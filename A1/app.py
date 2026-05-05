"""
中文词法分析 Web 应用
功能：文本规范化 / 多算法中文分词对比 / 词性标注（彩色展示）
"""

import re
import unicodedata
from collections import Counter

from flask import Flask, render_template, request, jsonify
import jieba
import jieba.posseg as pseg
import opencc

app = Flask(__name__)

# 初始化 OpenCC 繁体→简体转换器
t2s_converter = opencc.OpenCC("t2s")

# ---------- 加载 jieba 词典供 FMM/BMM 使用 ----------
jieba.initialize()
JIEBA_DICT = {word for word, freq in jieba.dt.FREQ.items() if freq > 0}
MAX_WORD_LEN = 16

# ---------- 停用词（词频统计时过滤） ----------
STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 被 从 把 让 与 向 对 当 年 月 日".split()
)


# ========== 文本规范化 ==========


def fullwidth_to_halfwidth(text: str) -> str:
    """全角字符转半角"""
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
    """
    文本规范化：
    1. 全角→半角
    2. Unicode NFC 归一化
    3. 去除特殊符号
    4. 繁体→简体
    """
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


# ========== 分词算法 ==========


def fmm_segment(text: str) -> list[str]:
    """正向最大匹配法 (Forward Maximum Matching)"""
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
    """逆向最大匹配法 (Backward Maximum Matching)"""
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
    """双向最大匹配法 (Bidirectional Maximum Matching)"""
    fmm_result = fmm_segment(text)
    bmm_result = bmm_segment(text)

    if fmm_result == bmm_result:
        return fmm_result

    # 启发式 1：词数更少者优先
    if len(fmm_result) != len(bmm_result):
        return fmm_result if len(fmm_result) < len(bmm_result) else bmm_result

    # 启发式 2：单字词更少者优先
    fmm_singles = sum(1 for w in fmm_result if len(w) == 1)
    bmm_singles = sum(1 for w in bmm_result if len(w) == 1)
    if fmm_singles != bmm_singles:
        return fmm_result if fmm_singles < bmm_singles else bmm_result

    # 默认取逆向结果
    return bmm_result


# ---------- 算法注册表 ----------
ALGORITHMS = {
    "fmm": ("正向最大匹配 (FMM)", fmm_segment),
    "bmm": ("逆向最大匹配 (BMM)", bmm_segment),
    "bimm": ("双向最大匹配 (BiMM)", bimm_segment),
    "jieba_precise": ("jieba 精确模式", lambda t: list(jieba.cut(t))),
    "jieba_full": ("jieba 全模式", lambda t: list(jieba.cut(t, cut_all=True))),
    "jieba_search": ("jieba 搜索引擎模式", lambda t: list(jieba.cut_for_search(t))),
}


def run_all_algorithms(text: str) -> dict:
    """对文本运行所有分词算法，返回各算法的结果与统计。"""
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
            "avg_length": round(
                sum(len(w) for w in words) / max(len(words), 1), 2
            ),
            "single_char_count": sum(1 for w in words if len(w) == 1),
            "freq_labels": [item[0] for item in freq],
            "freq_values": [item[1] for item in freq],
        }
    return results


def compute_comparison(all_results: dict) -> dict:
    """计算算法之间的对比统计。"""
    algo_ids = list(all_results.keys())
    all_word_sets = {k: set(v["words"]) for k, v in all_results.items()}

    # 两两 Jaccard 相似度
    pairwise = {}
    for i, a in enumerate(algo_ids):
        for b in algo_ids[i + 1 :]:
            inter = all_word_sets[a] & all_word_sets[b]
            union = all_word_sets[a] | all_word_sets[b]
            jaccard = len(inter) / len(union) if union else 1.0
            pairwise[f"{a}_vs_{b}"] = round(jaccard * 100, 1)

    # 各算法独有词
    unique_to = {}
    for aid in algo_ids:
        others = set()
        for oid in algo_ids:
            if oid != aid:
                others |= all_word_sets[oid]
        unique_to[aid] = sorted(all_word_sets[aid] - others)[:20]

    # 所有算法公共词
    common = set.intersection(*all_word_sets.values()) if all_word_sets else set()

    return {
        "pairwise_jaccard": pairwise,
        "unique_to": unique_to,
        "common_words": sorted(common)[:30],
        "common_count": len(common),
    }


# ========== 词性标注 ==========

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


def pos_tagging(text: str) -> list:
    """词性标注"""
    result = []
    for word, flag in pseg.cut(text):
        color, name = POS_COLOR_MAP.get(flag, DEFAULT_POS)
        result.append(
            {"word": word, "pos": flag, "color": color, "pos_name": name}
        )
    return result


# ========== 路由 ==========


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入文本"}), 400

    norm = normalize_text(text)
    cleaned = norm["simplified"]

    seg_results = run_all_algorithms(cleaned)
    comparison = compute_comparison(seg_results)
    pos = pos_tagging(cleaned)

    return jsonify({
        "normalize": norm,
        "segment": seg_results,
        "comparison": comparison,
        "pos": pos,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
