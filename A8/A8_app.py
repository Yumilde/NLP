import re

import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_nmt_pipeline():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")


def nmt_translate(text: str) -> str:
    translator = load_nmt_pipeline()
    result = translator(text.strip(), max_length=256)
    return result[0]["translation_text"] if result else ""


def safe_nmt_translate(text: str):
    try:
        return nmt_translate(text), None
    except Exception as exc:
        return None, str(exc)


RULE_DICT = {
    "it": "它", "is": "是", "rains": "下雨", "rain": "雨", "cats": "猫", "and": "和", "dogs": "狗",
    "machine": "机器", "translation": "翻译", "has": "已经", "evolved": "发展", "rapidly": "迅速地",
    "in": "在", "recent": "最近", "years": "年", "the": "这", "weather": "天气", "today": "今天",
    "was": "是", "very": "非常", "bad": "糟糕",
}


def rule_based_translate(text: str) -> str:
    tokens = text.strip().split()
    translated_tokens = []
    for token in tokens:
        match = re.match(r"^([A-Za-z']+)([^A-Za-z']*)$", token)
        if match:
            word = match.group(1)
            suffix = match.group(2)
            zh = RULE_DICT.get(word.lower(), word)
            translated_tokens.append(f"{zh}{suffix}")
        else:
            translated_tokens.append(token)
    return " ".join(translated_tokens)


def tokenize_zh(text: str):
    return [ch for ch in text.strip() if not ch.isspace()]


def bleu_interpretation(score: float) -> str:
    if score < 0.1: return "分数较低，候选译文与参考译文在词汇和短语层面的重合较少。"
    if score < 0.3: return "分数中等偏低，译文传达了部分信息，但在表达或细节上仍有明显差异。"
    if score < 0.5: return "分数中等，候选译文与参考译文有较多重合，整体可理解但仍可优化。"
    if score < 0.7: return "分数较高，候选译文与参考译文接近，翻译质量较好。"
    return "分数很高，候选译文与参考译文高度一致，自动评测表现优秀。"

def main():
    st.title("机器翻译对比与评测系统")
    st.caption("Week 9 随堂 Vibe 实验：机器翻译机制与质量评测")

    tab1, tab2, tab3 = st.tabs([
        "神经机器翻译引擎 (NMT Engine)",
        "直译引擎对比 (Rule-based)",
        "BLEU 自动评测",
    ])

    with tab1:
        st.subheader("模块 1：神经机器翻译引擎")
        en_text = st.text_area(
            "请输入英文句子",
            value="Natural language processing helps computers understand human language.",
            height=140,
            key="tab1_input"
        )
        if st.button("开始翻译", type="primary", use_container_width=True, key="tab1_btn"):
            if not en_text.strip():
                st.warning("请先输入英文句子。")
            else:
                with st.spinner("模型翻译中，请稍候..."):
                    zh_text, err_msg = safe_nmt_translate(en_text)
                if err_msg:
                    st.error(f"NMT 模型加载失败: {err_msg}")
                else:
                    st.success("翻译完成")
                    st.text_area("中文译文", value=zh_text, height=140, key="tab1_output")

    with tab2:
        st.subheader("模块 2：基于规则的直译 vs 神经网络意译")
        compare_text = st.text_area(
            "请输入用于对比的英文句子",
            value="It rains cats and dogs.",
            height=140,
            key="tab2_input"
        )
        if st.button("生成对比结果", type="primary", use_container_width=True, key="tab2_btn"):
            if not compare_text.strip():
                st.warning("请先输入英文句子。")
            else:
                with st.spinner("正在生成对比结果..."):
                    rule_text = rule_based_translate(compare_text)
                    nmt_text, err_msg = safe_nmt_translate(compare_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 基于规则的逐词直译")
                    st.text_area("直译结果", value=rule_text, height=180, key="tab2_rule_output")
                with col2:
                    st.markdown("### 神经网络意译 (NMT)")
                    st.text_area("NMT 结果", value=nmt_text, height=180, key="tab2_nmt_output")

    with tab3:
        st.subheader("模块 3：机器翻译质量自动评测 (BLEU Score)")
        bleu_source_en = st.text_area(
            "1. 待翻译英文原文",
            value="The weather today is very good.",
            height=100,
            key="tab3_source"
        )
        if st.button("调用 NMT 生成候选译文", use_container_width=True, key="tab3_gen_btn"):
            if not bleu_source_en.strip():
                st.warning("请先输入英文原文。")
            else:
                with st.spinner("正在调用 NMT 生成候选译文..."):
                    candidate_text, err_msg = safe_nmt_translate(bleu_source_en)
                if err_msg:
                    st.error(f"NMT 模型加载失败: {err_msg}")
                else:
                    st.session_state["tab3_candidate"] = candidate_text
                    st.success("候选译文已生成并填入下方输入框。")

        reference_zh = st.text_area(
            "2. 标准中文参考译文 (Reference)",
            value="今天天气非常好。",
            height=100,
            key="tab3_reference"
        )
        candidate_zh = st.text_area("3. 机器候选译文 (Candidate)", value=st.session_state.get("tab3_candidate", ""), height=100, key="tab3_candidate_area")

        if st.button("计算 BLEU 分数", type="primary", use_container_width=True, key="tab3_bleu_btn"):
            if not reference_zh.strip() or not candidate_zh.strip():
                st.warning("请完整填写参考译文和候选译文。")
            else:
                ref_tokens = tokenize_zh(reference_zh)
                cand_tokens = tokenize_zh(candidate_zh)
                if not ref_tokens or not cand_tokens:
                    st.error("输入内容过短，无法计算 BLEU。")
                else:
                    smoothie = SmoothingFunction().method1
                    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
                    st.metric("BLEU 得分", f"{bleu_score:.4f}")
                    st.write(f"结果解读：{bleu_interpretation(bleu_score)}")

main()
