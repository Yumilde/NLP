import re
import unicodedata
from collections import Counter

import jieba
import jieba.posseg as pseg
import opencc
import pandas as pd
import streamlit as st

st.set_page_config(page_title="A1 中文词法分析", layout="wide")

jieba.initialize()
JIEBA_DICT = {word for word, freq in jieba.dt.FREQ.items() if freq > 0}
MAX_WORD_LEN = 16
STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 被 从 把 让 与 向 对 当 年 月 日".split()
)
POS_COLOR_MAP = {
    "n": "名词", "ns": "地名", "nr": "人名", "nt": "机构名", "nz": "专名", "v": "动词", "a": "形容词", "d": "副词",
    "m": "数词", "q": "量词", "r": "代词", "p": "介词", "c": "连词", "u": "助词", "x": "标点", "w": "标点",
}


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
    t2s = opencc.OpenCC("t2s")
    step1 = fullwidth_to_halfwidth(text)
    step2 = unicodedata.normalize("NFC", step1)
    step3 = re.sub(
        r"[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9，。！？、；：\u201c\u201d\u2018\u2019（）《》【】\-\s,.!?;:\"'()\[\]]",
        "",
        step2,
    )
    step4 = t2s.convert(step3)
    return {"原文": text, "全半角归一": step1, "清洗后": step3, "简体化": step4}


def fmm_segment(text: str) -> list[str]:
    words, i, n = [], 0, len(text)
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
    words, i = [], len(text)
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
    fmm = fmm_segment(text)
    bmm = bmm_segment(text)
    if fmm == bmm:
        return fmm
    if len(fmm) != len(bmm):
        return fmm if len(fmm) < len(bmm) else bmm
    fmm_singles = sum(1 for w in fmm if len(w) == 1)
    bmm_singles = sum(1 for w in bmm if len(w) == 1)
    if fmm_singles != bmm_singles:
        return fmm if fmm_singles < bmm_singles else bmm
    return bmm


def run_all_algorithms(text: str) -> dict:
    algos = {
        "FMM": fmm_segment,
        "BMM": bmm_segment,
        "BiMM": bimm_segment,
        "jieba精确": lambda t: list(jieba.cut(t)),
        "jieba全模式": lambda t: list(jieba.cut(t, cut_all=True)),
        "jieba搜索": lambda t: list(jieba.cut_for_search(t)),
    }
    results = {}
    for name, fn in algos.items():
        words = fn(text)
        filtered = [w for w in words if len(w) >= 2 and w not in STOP_WORDS]
        freq = Counter(filtered).most_common(10)
        results[name] = {
            "words": words,
            "word_count": len(words),
            "unique_count": len(set(words)),
            "single_count": sum(1 for w in words if len(w) == 1),
            "freq": freq,
        }
    return results


def pos_tagging(text: str):
    rows = []
    for w, p in pseg.cut(text):
        rows.append({"词": w, "词性": p, "词性说明": POS_COLOR_MAP.get(p, "其他")})
    return rows


st.title("A1：中文词法分析")

st.markdown("### 实验任务")
st.write("完成中文文本规范化、分词算法对比（FMM/BMM/BiMM/jieba）与词性标注可视化。")

st.markdown("### 可交互演示")
text = st.text_area(
    "输入中文文本",
    value="我来到上海交通大学自然语言处理课堂，今天我们一起做vibe coding作业展示。",
    height=120,
)

if st.button("运行 A1 分析", type="primary"):
    if not text.strip():
        st.warning("请输入文本后再运行。")
    else:
        normalized = normalize_text(text)
        norm_text = normalized["简体化"]

        st.markdown("#### 文本规范化")
        st.json(normalized)

        st.markdown("#### 分词算法对比")
        results = run_all_algorithms(norm_text)
        table = []
        for name, item in results.items():
            table.append(
                {
                    "算法": name,
                    "词数": item["word_count"],
                    "去重词数": item["unique_count"],
                    "单字词数": item["single_count"],
                    "Top词频": ", ".join([f"{w}:{c}" for w, c in item["freq"][:5]]) or "-",
                }
            )
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        with st.expander("查看各算法分词序列"):
            for name, item in results.items():
                st.write(f"**{name}**")
                st.code(" / ".join(item["words"]), language="text")

        st.markdown("#### 词性标注")
        st.dataframe(pd.DataFrame(pos_tagging(norm_text)), use_container_width=True, hide_index=True)

st.markdown("### 结果对比与结论")
st.info("FMM/BMM/BiMM 体现规则法差异，jieba 系列体现统计词典优势；课堂分享建议重点展示‘同一句子在不同算法下的切分边界变化’。")
