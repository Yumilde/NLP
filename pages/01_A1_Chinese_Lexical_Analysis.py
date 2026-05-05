import re
import unicodedata
from collections import Counter

import jieba
import jieba.posseg as pseg
import opencc
import pandas as pd
import streamlit as st

st.set_page_config(page_title="A1 中文词法分析", layout="wide")

# Init once for session
jieba.initialize()
t2s_converter = opencc.OpenCC("t2s")
JIEBA_DICT = {word for word, freq in jieba.dt.FREQ.items() if freq > 0}
MAX_WORD_LEN = 16

STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 被 从 把 让 与 向 对 当 年 月 日".split()
)

POS_COLOR_MAP = {
    "n": ("#e74c3c", "名词"),
    "ns": ("#e74c3c", "地名"),
    "nr": ("#e74c3c", "人名"),
    "nt": ("#e74c3c", "机构名"),
    "nz": ("#e74c3c", "其他专名"),
    "v": ("#2980b9", "动词"),
    "vd": ("#2980b9", "副动词"),
    "vn": ("#8e44ad", "名动词"),
    "a": ("#27ae60", "形容词"),
    "ad": ("#27ae60", "副形词"),
    "an": ("#27ae60", "名形词"),
    "d": ("#f39c12", "副词"),
    "m": ("#1abc9c", "数词"),
    "q": ("#1abc9c", "量词"),
    "r": ("#9b59b6", "代词"),
    "p": ("#95a5a6", "介词"),
    "c": ("#95a5a6", "连词"),
    "u": ("#bdc3c7", "助词"),
    "x": ("#bdc3c7", "标点"),
    "w": ("#bdc3c7", "标点"),
}
DEFAULT_POS = ("#7f8c8d", "其他")


def fullwidth_to_halfwidth(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            result.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(ch)
    return "".join(result)


def normalize_text(text: str) -> dict:
    step1 = fullwidth_to_halfwidth(text)
    step2 = unicodedata.normalize("NFC", step1)
    step3 = re.sub(
        r"[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9"
        r"，。！？、；：\u201c\u201d\u2018\u2019（）《》【】\-\s,.!?;:\"'()\[\]]",
        "",
        step2,
    )
    step4 = t2s_converter.convert(step3)
    return {
        "original": text,
        "half_width": step1,
        "clean": step3,
        "simplified": step4,
    }


def fmm_segment(text: str) -> list[str]:
    words = []
    i = 0
    n = len(text)
    while i < n:
        max_end = min(i + MAX_WORD_LEN, n)
        matched = False
        for end in range(max_end, i + 1, -1):
            candidate = text[i:end]
            if candidate in JIEBA_DICT:
                words.append(candidate)
                i = end
                matched = True
                break
        if not matched:
            words.append(text[i])
            i += 1
    return words


def bmm_segment(text: str) -> list[str]:
    words = []
    i = len(text)
    while i > 0:
        max_start = max(i - MAX_WORD_LEN, 0)
        matched = False
        for start in range(max_start, i):
            candidate = text[start:i]
            if candidate in JIEBA_DICT:
                words.append(candidate)
                i = start
                matched = True
                break
        if not matched:
            words.append(text[i - 1])
            i -= 1
    words.reverse()
    return words


def bimm_segment(text: str) -> list[str]:
    fmm_result = fmm_segment(text)
    bmm_result = bmm_segment(text)
    if fmm_result == bmm_result:
        return fmm_result
    if len(fmm_result) != len(bmm_result):
        return fmm_result if len(fmm_result) < len(bmm_result) else bmm_result
    fmm_singles = sum(1 for w in fmm_result if len(w) == 1)
    bmm_singles = sum(1 for w in bmm_result if len(w) == 1)
    if fmm_singles != bmm_singles:
        return fmm_result if fmm_singles < bmm_singles else bmm_result
    return bmm_result


ALGORITHMS = {
    "fmm": ("正向最大匹配 (FMM)", fmm_segment),
    "bmm": ("逆向最大匹配 (BMM)", bmm_segment),
    "bimm": ("双向最大匹配 (BiMM)", bimm_segment),
    "jieba_precise": ("jieba 精确模式", lambda t: list(jieba.cut(t))),
    "jieba_full": ("jieba 全模式", lambda t: list(jieba.cut(t, cut_all=True))),
    "jieba_search": ("jieba 搜索引擎模式", lambda t: list(jieba.cut_for_search(t))),
}


def run_all_algorithms(text: str) -> dict:
    results = {}
    for algo_id, (algo_name, algo_fn) in ALGORITHMS.items():
        words = algo_fn(text)
        filtered = [w for w in words if len(w) >= 2 and w not in STOP_WORDS]
        freq = Counter(filtered).most_common(10)
        results[algo_id] = {
            "name": algo_name,
            "words": words,
            "word_count": len(words),
            "unique_count": len(set(words)),
            "avg_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
            "single_char_count": sum(1 for w in words if len(w) == 1),
            "freq_labels": [item[0] for item in freq],
            "freq_values": [item[1] for item in freq],
        }
    return results


def compute_comparison(all_results: dict) -> dict:
    algo_ids = list(all_results.keys())
    all_word_sets = {k: set(v["words"]) for k, v in all_results.items()}

    pairwise = {}
    for i, a in enumerate(algo_ids):
        for b in algo_ids[i + 1 :]:
            inter = all_word_sets[a] & all_word_sets[b]
            union = all_word_sets[a] | all_word_sets[b]
            jaccard = len(inter) / len(union) if union else 1.0
            pairwise[f"{a}_vs_{b}"] = round(jaccard * 100, 1)

    unique_to = {}
    for aid in algo_ids:
        others = set()
        for oid in algo_ids:
            if oid != aid:
                others |= all_word_sets[oid]
        unique_to[aid] = sorted(all_word_sets[aid] - others)[:20]

    common = set.intersection(*all_word_sets.values()) if all_word_sets else set()

    return {
        "pairwise_jaccard": pairwise,
        "unique_to": unique_to,
        "common_words": sorted(common)[:30],
        "common_count": len(common),
    }


def pos_tagging(text: str) -> list:
    result = []
    for word, flag in pseg.cut(text):
        color, name = POS_COLOR_MAP.get(flag, DEFAULT_POS)
        result.append({"word": word, "pos": flag, "color": color, "pos_name": name})
    return result


def render_pos_html(pos_list: list[dict]) -> str:
    spans = []
    for item in pos_list:
        w = item["word"]
        color = item["color"]
        name = item["pos_name"]
        spans.append(
            f"<span style='display:inline-block;margin:3px;padding:4px 8px;border-radius:8px;"
            f"background:{color};color:white;font-size:14px;' title='{name}'>"
            f"{w}<small style='margin-left:6px;opacity:0.85;'>/{item['pos']}</small></span>"
        )
    return "".join(spans)


st.title("A1：中文词法分析平台（Streamlit 版）")
st.caption("文本规范化 + 多算法分词对比 + 词性标注可视化")

text = st.text_area(
    "输入中文文本",
    value="今天天气不错，我想去外滩散步，然后和同学讨论自然语言处理课程作业。",
    height=140,
)

if st.button("开始分析", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("请输入文本。")
    else:
        norm = normalize_text(text)
        cleaned = norm["simplified"]

        st.subheader("1) 文本规范化")
        st.json(norm)

        st.subheader("2) 分词结果对比")
        seg_results = run_all_algorithms(cleaned)
        comparison = compute_comparison(seg_results)

        metrics = []
        for aid, info in seg_results.items():
            metrics.append(
                {
                    "算法": info["name"],
                    "词数": info["word_count"],
                    "去重词数": info["unique_count"],
                    "平均词长": info["avg_length"],
                    "单字词数": info["single_char_count"],
                }
            )
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

        algo_choice = st.selectbox(
            "查看某个分词算法的详细分词序列",
            options=list(seg_results.keys()),
            format_func=lambda x: seg_results[x]["name"],
        )
        st.code(" / ".join(seg_results[algo_choice]["words"]))

        st.subheader("3) 算法相似度（Jaccard %）")
        jac_rows = [{"算法对": k, "Jaccard(%)": v} for k, v in comparison["pairwise_jaccard"].items()]
        st.dataframe(pd.DataFrame(jac_rows), use_container_width=True, hide_index=True)

        st.subheader("4) 词性标注")
        pos = pos_tagging(cleaned)
        st.markdown(render_pos_html(pos), unsafe_allow_html=True)

        pos_rows = [{"词": i["word"], "词性": i["pos"], "类别": i["pos_name"]} for i in pos]
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
