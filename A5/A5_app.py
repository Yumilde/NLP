import html
import re
from typing import List, Optional, Tuple

import requests
import spacy
import streamlit as st
from spacy.cli import download as spacy_download

try:
    from fastcoref import FCoref
    import_error = None
except Exception as e:
    FCoref = None
    import_error = str(e)


NEURALEDUSEG_BASE = (
    "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING"
)
SAMPLE_FILES = [
    "wsj_0601.out",
    "wsj_0604.out",
    "wsj_0615.out",
]
FALLBACK_SAMPLE_TEXT = (
    "The company released a new product, and customers responded positively. "
    "Although the launch faced delays, the team solved most issues quickly. "
    "Because demand increased, the firm expanded its support service."
)
FALLBACK_SAMPLE_EDUS = [
    "The company released a new product,",
    "and customers responded positively.",
    "Although the launch faced delays,",
    "the team solved most issues quickly.",
    "Because demand increased,",
    "the firm expanded its support service.",
]

PDTB_CONNECTIVES = {
    "when": "TEMPORAL",
    "after": "TEMPORAL",
    "before": "TEMPORAL",
    "while": "TEMPORAL",
    "until": "TEMPORAL",
    "because": "CONTINGENCY",
    "since": "CONTINGENCY",
    "if": "CONTINGENCY",
    "therefore": "CONTINGENCY",
    "thus": "CONTINGENCY",
    "but": "COMPARISON",
    "although": "COMPARISON",
    "though": "COMPARISON",
    "however": "COMPARISON",
    "whereas": "COMPARISON",
    "and": "EXPANSION",
    "or": "EXPANSION",
    "also": "EXPANSION",
    "moreover": "EXPANSION",
    "in addition": "EXPANSION",
}


@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception:
        return None


@st.cache_resource
def load_fastcoref_model():
    if FCoref is None:
        return None

    # 用 CPU 保证在普通课堂环境可运行；若用户有 GPU 可按需改为 cuda。
    return FCoref(device="cpu")


@st.cache_data(show_spinner=False)
def fetch_neuraleduseg_sample(sample_file: str) -> Tuple[str, str]:
    text_url = f"{NEURALEDUSEG_BASE}/{sample_file}"
    edu_url = f"{NEURALEDUSEG_BASE}/{sample_file}.edus"

    text_resp = requests.get(text_url, timeout=10)
    edu_resp = requests.get(edu_url, timeout=10)
    text_resp.raise_for_status()
    edu_resp.raise_for_status()

    return text_resp.text, edu_resp.text


def parse_ground_truth_edus(edus_text: str, max_edus: int) -> List[str]:
    lines = [line.strip() for line in edus_text.splitlines() if line.strip()]
    return lines[:max_edus]


def baseline_segment_with_spacy(text: str, nlp, max_edus: int) -> List[str]:
    doc = nlp(text)
    edus: List[str] = []
    current: List[str] = []

    subordinators = {
        "because",
        "although",
        "though",
        "while",
        "if",
        "when",
        "since",
        "unless",
        "whereas",
        "that",
    }

    for token in doc:
        if token.is_space:
            continue

        split_before = (
            len(current) >= 4
            and token.lower_ in subordinators
            and token.pos_ == "SCONJ"
        ) or (len(current) >= 5 and token.dep_ in {"advcl", "relcl"} and token.pos_ in {"VERB", "AUX"})

        if split_before:
            edus.append(" ".join(current).strip())
            current = []

        current.append(token.text)

        split_after = token.text in {".", "!", "?", ";"} or (
            token.text == "," and len(current) >= 8
        )

        if split_after:
            edus.append(" ".join(current).strip())
            current = []

        if len(edus) >= max_edus:
            break

    if current and len(edus) < max_edus:
        edus.append(" ".join(current).strip())

    return [edu for edu in edus if edu]


def highlight_boundary_token(edu: str) -> str:
    tokens = re.findall(r"\S+", edu)
    if not tokens:
        return ""

    safe_tokens = [html.escape(tok) for tok in tokens]
    safe_tokens[-1] = (
        f"<span style='background:#ffe08a;padding:1px 4px;border-radius:4px;'>"
        f"{safe_tokens[-1]}</span>"
    )
    return " ".join(safe_tokens)


def render_edu_cards(edus: List[str], card_border: str) -> None:
    card_template = (
        "<div style='border:1px solid {border};border-radius:10px;padding:10px 12px;"
        "margin-bottom:10px;background:#fff;'>"
        "<div style='font-size:12px;color:#555;margin-bottom:6px;'>EDU {idx}</div>"
        "<div style='line-height:1.7;'>{content}</div>"
        "</div>"
    )
    for i, edu in enumerate(edus, start=1):
        st.markdown(
            card_template.format(
                border=card_border,
                idx=i,
                content=highlight_boundary_token(edu),
            ),
            unsafe_allow_html=True,
        )


def find_explicit_connectives(sentence: str) -> List[Tuple[int, int, str, str]]:
    matches: List[Tuple[int, int, str, str]] = []
    for conn, relation in PDTB_CONNECTIVES.items():
        # 修正：使用 r"\b" 表示正则表达式中的单词边界
        pattern = r"\b" + re.escape(conn) + r"\b"
        for m in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), m.group(0), relation))
    matches.sort(key=lambda x: x[0])
    return matches


def highlight_connectives(sentence: str, matches: List[Tuple[int, int, str, str]]) -> str:
    if not matches:
        return html.escape(sentence)

    color_map = {
        "TEMPORAL": "#2e86de",
        "CONTINGENCY": "#16a085",
        "COMPARISON": "#e67e22",
        "EXPANSION": "#8e44ad",
    }

    out_parts: List[str] = []
    cursor = 0
    for start, end, conn, relation in matches:
        if start < cursor:
            continue

        out_parts.append(html.escape(sentence[cursor:start]))
        color = color_map.get(relation, "#c0392b")
        label = f"{conn} [{relation}]"
        out_parts.append(
            "<strong style='color:{};background:#fff3cd;padding:1px 4px;border-radius:4px;'>".format(color)
            + html.escape(label)
            + "</strong>"
        )
        cursor = end

    out_parts.append(html.escape(sentence[cursor:]))
    return "".join(out_parts)


def split_args_by_connective(sentence: str, match: Tuple[int, int, str, str]) -> Tuple[str, str, str, str]:
    start, end, conn, relation = match
    arg1 = sentence[:start].strip(" ,;-\n\t")
    arg2 = sentence[end:].strip(" ,;-\n\t")
    return arg1, arg2, conn, relation


def render_argument_box(title: str, text: str, bg_color: str) -> None:
    safe_text = html.escape(text) if text else "(空)"
    st.markdown(
        (
            "<div style='border:1px solid #ddd;border-radius:10px;padding:10px 12px;"
            "background:{};margin-bottom:8px;'>"
            "<div style='font-size:12px;color:#555;margin-bottom:6px;'><strong>{}</strong></div>"
            "<div style='line-height:1.7;'>{}</div>"
            "</div>"
        ).format(bg_color, html.escape(title), safe_text),
        unsafe_allow_html=True,
    )


def extract_coref_clusters(text: str, predictor) -> List[List[str]]:
    if predictor is None:
        return []

    predictions = predictor.predict(texts=[text])
    if not predictions:
        return []

    result = predictions[0]
    clusters_raw = []

    if hasattr(result, "get_clusters"):
        try:
            clusters_raw = result.get_clusters(as_strings=True)
        except TypeError:
            clusters_raw = result.get_clusters()
    elif hasattr(result, "clusters"):
        clusters_raw = result.clusters

    normalized: List[List[str]] = []
    for cluster in clusters_raw:
        cluster_mentions: List[str] = []
        for mention in cluster:
            if isinstance(mention, str):
                m = mention.strip()
            else:
                m = str(mention).strip()

            if m and m not in cluster_mentions:
                cluster_mentions.append(m)

        if len(cluster_mentions) >= 2:
            normalized.append(cluster_mentions)

    return normalized


def render_coref_highlight(text: str, clusters: List[List[str]]) -> str:
    if not text.strip() or not clusters:
        return html.escape(text)

    palette = [
        "#fde68a",
        "#bfdbfe",
        "#bbf7d0",
        "#fecaca",
        "#ddd6fe",
        "#fbcfe8",
        "#a7f3d0",
        "#fdba74",
    ]

    candidates = []
    for cid, cluster in enumerate(clusters):
        color = palette[cid % len(palette)]
        for mention in cluster:
            if len(mention.strip()) <= 1:
                continue
            for m in re.finditer(re.escape(mention), text):
                candidates.append(
                    {
                        "start": m.start(),
                        "end": m.end(),
                        "color": color,
                        "mention": mention,
                        "cluster": cid + 1,
                    }
                )

    if not candidates:
        return html.escape(text)

    candidates.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    selected = []
    last_end = -1
    for span in candidates:
        if span["start"] >= last_end:
            selected.append(span)
            last_end = span["end"]

    out = []
    cursor = 0
    for span in selected:
        s = span["start"]
        e = span["end"]
        out.append(html.escape(text[cursor:s]))
        out.append(
            (
                "<span style='background:{};padding:1px 4px;border-radius:4px;' "
                "title='Cluster {}'>{}</span>"
            ).format(
                span["color"],
                span["cluster"],
                html.escape(text[s:e]),
            )
        )
        cursor = e

    out.append(html.escape(text[cursor:]))
    return "".join(out)


def tab_edu_segmentation() -> None:
    st.subheader("模块 1：话语分割（EDU 切分）")
    st.caption("规则基线 vs NeuralEDUSeg 真实标注")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        sample_file = st.selectbox("选择 NeuralEDUSeg 样本文件", SAMPLE_FILES, index=0)
    with col_b:
        max_edus = st.slider("展示 EDU 数量", min_value=5, max_value=40, value=16, step=1)

    try:
        raw_text, edus_text = fetch_neuraleduseg_sample(sample_file)
    except Exception as exc:
        st.warning(f"无法获取远程数据，已切换到内置样例：{exc}")
        raw_text = FALLBACK_SAMPLE_TEXT
        edus_text = "\n".join(FALLBACK_SAMPLE_EDUS)

    gt_edus = parse_ground_truth_edus(edus_text, max_edus=max_edus)

    nlp = load_spacy_model()
    baseline_edus = baseline_segment_with_spacy(raw_text, nlp=nlp, max_edus=max_edus)

    st.markdown("### 原始文本片段")
    st.text_area("来自 NeuralEDUSeg 的原始文本（.out）", raw_text[:2500], height=200)

    st.markdown("### 切分对比（边界词已高亮）")
    left, right = st.columns(2)

    with left:
        st.markdown("#### 规则基线切分结果")
        st.caption("启发式规则：标点、SCONJ 从属连词、部分从句相关依存模式")
        render_edu_cards(baseline_edus, card_border="#8fb6ff")

    with right:
        st.markdown("#### NeuralEDUSeg 数据集真实标注")
        st.caption("来自 .edus 文件的 Ground Truth EDU")
        render_edu_cards(gt_edus, card_border="#7ccf9d")

    st.markdown("### 边界词列表（用于模拟序列标注观察）")
    col1, col2 = st.columns(2)
    with col1:
        baseline_boundary_tokens = [re.findall(r"\S+", x)[-1] for x in baseline_edus if re.findall(r"\S+", x)]
        st.write("规则基线边界词：", baseline_boundary_tokens)
    with col2:
        gt_boundary_tokens = [re.findall(r"\S+", x)[-1] for x in gt_edus if re.findall(r"\S+", x)]
        st.write("真实标注边界词：", gt_boundary_tokens)


def tab_shallow_discourse() -> None:
    st.subheader("模块 2：浅层篇章分析与显式关系提取")
    st.caption("PDTB 显式连接词识别 + Arg1/Arg2 简易抽取")

    default_text = (
        "Third-quarter sales in Europe were exceptionally strong, boosted by promotional "
        "programs and new products - although weaker foreign currencies reduced the company's "
        "earnings."
    )
    sentence = st.text_area("输入句子", value=default_text, height=130)

    matches = find_explicit_connectives(sentence)

    st.markdown("### 显式连接词标注")
    highlighted = highlight_connectives(sentence, matches)
    st.markdown(
        (
            "<div style='border:1px solid #d8d8d8;border-radius:10px;padding:10px 12px;"
            "background:#ffffff;line-height:1.8;'>"
            "{}"
            "</div>"
        ).format(highlighted),
        unsafe_allow_html=True,
    )

    if not matches:
        st.warning("未匹配到预设显式连接词。可尝试 because / although / but / and / when 等。")
        return

    st.markdown("### 连接词识别结果")
    for _, _, conn, relation in matches:
        st.write(f"- {conn} [{relation}]")

    st.markdown("### Arg1 / Arg2 简易切分")
    selected = matches[0]
    arg1, arg2, conn, relation = split_args_by_connective(sentence, selected)
    st.caption(f"当前以第一个连接词 '{conn}' 作为切分点，关系类别：{relation}")

    col_left, col_right = st.columns(2)
    with col_left:
        render_argument_box("Arg1（前置论据）", arg1, "#e8f4ff")
    with col_right:
        render_argument_box("Arg2（后置论据）", arg2, "#ecfff1")


def tab_coreference() -> None:
    st.subheader("模块 3：指代消解（Coreference Resolution）可视化")
    st.caption("fastcoref 端到端指代聚类 + 原文高亮")

    default_paragraph = (
        "Barack Obama was born in Hawaii. He served as the 44th President of the United States. "
        "Obama studied at Harvard Law School, where he became the president of the law review. "
        "Later, he wrote a memoir about his early life, and his book became widely known."
    )
    text = st.text_area("输入英文段落", value=default_paragraph, height=180)

    predictor = load_fastcoref_model()
    if predictor is None:
        st.error(f"未检测到 fastcoref 或加载失败。错误信息: {import_error}")
        st.info("请尝试在本地环境执行：pip install fastcoref")
        return

    if not text.strip():
        st.warning("请输入待分析文本。")
        return

    try:
        clusters = extract_coref_clusters(text, predictor)
    except Exception as exc:
        st.error(f"指代消解运行失败：{exc}")
        return

    highlighted = render_coref_highlight(text, clusters)
    st.markdown("### 原文高亮（同一实体同色）")
    st.markdown(
        (
            "<div style='border:1px solid #d8d8d8;border-radius:10px;padding:12px;"
            "background:#ffffff;line-height:1.9;'>"
            "{}"
            "</div>"
        ).format(highlighted),
        unsafe_allow_html=True,
    )

    st.markdown("### Coreference Clusters")
    if not clusters:
        st.info("未提取到明显的指代簇，可尝试输入含更多代词（he/she/they/his/her/their）的文本。")
        return

    for i, cluster in enumerate(clusters, start=1):
        st.write(f"Cluster {i}: {cluster}")


def main() -> None:
    st.title("构建'篇章分析综合平台'")
    st.caption("Week 6 随堂 Vibe 实验：篇章分析与指代消解系统")
    tab1, tab2, tab3 = st.tabs(
        [
            "模块 1：EDU 切分对比",
            "模块 2：浅层篇章关系提取",
            "模块 3：指代消解可视化",
        ]
    )

    with tab1:
        tab_edu_segmentation()

    with tab2:
        tab_shallow_discourse()

    with tab3:
        tab_coreference()


main()
