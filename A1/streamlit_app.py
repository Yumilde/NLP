import streamlit as st
import re
import unicodedata
from collections import Counter
import jieba
import jieba.posseg as pseg
import opencc

# 初始化 OpenCC 繁体→简体转换器
@st.cache_resource
def get_converter():
    return opencc.OpenCC("t2s")

t2s_converter = get_converter()

# ---------- 加载 jieba 词典供 FMM/BMM 使用 ----------
jieba.initialize()
JIEBA_DICT = {word for word, freq in jieba.dt.FREQ.items() if freq > 0}
MAX_WORD_LEN = 16

# ---------- 停用词（词频统计时过滤） ----------
STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 被 从 把 让 与 向 对 当 年 月 日".split()
)

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
    n = len(text)
    i = n
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
    if fmm_result == bmm_result: return fmm_result
    if len(fmm_result) != len(bmm_result):
        return fmm_result if len(fmm_result) < len(bmm_result) else bmm_result
    fmm_singles = sum(1 for w in fmm_result if len(w) == 1)
    bmm_singles = sum(1 for w in bmm_result if len(w) == 1)
    return fmm_result if fmm_singles < bmm_singles else bmm_result

ALGORITHMS = {
    "fmm": ("正向最大匹配 (FMM)", fmm_segment),
    "bmm": ("逆向最大匹配 (BMM)", bmm_segment),
    "bimm": ("双向最大匹配 (BiMM)", bimm_segment),
    "jieba_precise": ("jieba 精确模式", lambda t: list(jieba.cut(t))),
}

def run_all_algorithms(text: str) -> dict:
    results = {}
    for algo_id, (algo_name, algo_fn) in ALGORITHMS.items():
        words = algo_fn(text)
        results[algo_id] = {"name": algo_name, "words": words}
    return results

POS_COLOR_MAP = {
    "n": "#e74c3c", "ns": "#e74c3c", "nr": "#e74c3c", "v": "#2980b9", "a": "#27ae60", "d": "#f39c12"
}

def main():
    st.title("A1: 中文词法分析")
    text = st.text_area("输入中文文本", "北京欢迎你！他来到了清华大学。")
    if st.button("开始分析", type="primary"):
        norm = normalize_text(text)
        st.subheader("1. 文本规范化")
        st.json(norm)
        
        st.subheader("2. 分词算法对比")
        results = run_all_algorithms(norm["simplified"])
        for algo_id, data in results.items():
            st.write(f"**{data['name']}**: {' / '.join(data['words'])}")
            
        st.subheader("3. 词性标注")
        words = pseg.cut(norm["simplified"])
        html_str = ""
        for word, flag in words:
            color = POS_COLOR_MAP.get(flag[0], "#7f8c8d") if flag else "#7f8c8d"
            html_str += f'<span style="color:{color}; margin-right:5px;">{word}({flag})</span>'
        st.markdown(html_str, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
