import html
import json
import re
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components

Entity = Dict[str, object]
Relation = Dict[str, str]


# Mock/规则混合实体词典，后续可替换为 spaCy 或大模型 API。
ENTITY_LEXICON: Dict[str, str] = {
    "Steve Jobs": "PER",
    "Tim Cook": "PER",
    "马云": "PER",
    "雷军": "PER",
    "Apple": "ORG",
    "Microsoft": "ORG",
    "阿里巴巴": "ORG",
    "小米": "ORG",
    "California": "LOC",
    "Beijing": "LOC",
    "杭州": "LOC",
    "北京": "LOC",
}

COLOR_MAP = {
    "PER": "#ffe0b2",
    "ORG": "#c8e6c9",
    "LOC": "#bbdefb",
}

LABEL_NAME_MAP = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
}


def find_entities(text: str) -> List[Entity]:
    candidates: List[Entity] = []

    for phrase, label in ENTITY_LEXICON.items():
        if not phrase:
            continue

        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            candidates.append(
                {
                    "text": text[match.start() : match.end()],
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # 平铺实体去重：优先保留更长实体，避免重叠高亮。
    candidates.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))

    selected: List[Entity] = []
    occupied_ranges: List[Tuple[int, int]] = []

    for entity in candidates:
        start, end = int(entity["start"]), int(entity["end"])
        overlap = any(not (end <= s or start >= e) for s, e in occupied_ranges)
        if overlap:
            continue

        selected.append(entity)
        occupied_ranges.append((start, end))

    selected.sort(key=lambda item: item["start"])
    return selected


def extract_relations(text: str, entities: List[Entity]) -> List[Relation]:
    relations: List[Relation] = []
    unique = set()

    person_entities = [e for e in entities if str(e["label"]) == "PER"]
    org_entities = [e for e in entities if str(e["label"]) == "ORG"]
    loc_entities = [e for e in entities if str(e["label"]) == "LOC"]

    founder_hint = re.search(r"创立|创办|创建|founded|founder", text, flags=re.IGNORECASE)
    located_hint = re.search(r"位于|总部|located in|headquartered in|based in", text, flags=re.IGNORECASE)

    if founder_hint:
        for person in person_entities:
            for org in org_entities:
                triple = (str(person["text"]), "FOUNDER_OF", str(org["text"]))
                if triple in unique:
                    continue
                unique.add(triple)
                relations.append(
                    {
                        "source": triple[0],
                        "relation": triple[1],
                        "target": triple[2],
                    }
                )

    if located_hint:
        for org in org_entities:
            for loc in loc_entities:
                triple = (str(org["text"]), "LOCATED_IN", str(loc["text"]))
                if triple in unique:
                    continue
                unique.add(triple)
                relations.append(
                    {
                        "source": triple[0],
                        "relation": triple[1],
                        "target": triple[2],
                    }
                )

    return relations


def extract_info(text: str) -> Dict[str, List[Dict[str, object]]]:
    entities = find_entities(text)
    relations = extract_relations(text, entities)
    return {
        "entities": entities,
        "relations": relations,
    }


def convert_to_graph_data(entities: List[Entity], relations: List[Relation]) -> Dict[str, List[Dict[str, object]]]:
        node_style = {
                "PER": {"color": "#ffb74d", "size": 30},
                "ORG": {"color": "#81c784", "size": 34},
                "LOC": {"color": "#64b5f6", "size": 28},
        }

        nodes: List[Dict[str, object]] = []
        edges: List[Dict[str, object]] = []
        node_id_map: Dict[Tuple[str, str], str] = {}
        text_to_node_id: Dict[str, str] = {}

        for entity in entities:
                text = str(entity["text"])
                label = str(entity["label"])
                key = (text, label)
                if key in node_id_map:
                        continue

                node_id = f"{label}:{text}"
                style = node_style.get(label, {"color": "#bdbdbd", "size": 26})
                nodes.append(
                        {
                                "id": node_id,
                                "label": text,
                                "title": f"{text} ({LABEL_NAME_MAP.get(label, label)})",
                                "group": label,
                                "color": style["color"],
                                "size": style["size"],
                        }
                )
                node_id_map[key] = node_id
                if text not in text_to_node_id:
                        text_to_node_id[text] = node_id

        for relation in relations:
                source_text = str(relation["source"])
                target_text = str(relation["target"])
                relation_text = str(relation["relation"])

                source_id = text_to_node_id.get(source_text)
                target_id = text_to_node_id.get(target_text)

                if not source_id or not target_id:
                        continue

                edges.append(
                        {
                                "from": source_id,
                                "to": target_id,
                                "label": relation_text,
                                "arrows": "to",
                        }
                )

        return {"nodes": nodes, "edges": edges}


def render_knowledge_graph(nodes: List[Dict[str, object]], edges: List[Dict[str, object]]) -> None:
        vis_data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

        graph_html = f"""
        <div id="kg-graph" style="height: 520px; border: 1px solid #d9d9d9; border-radius: 10px; background: #fff;"></div>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <script>
            (function() {{
                const dataObj = {vis_data};
                const container = document.getElementById('kg-graph');
                const data = {{
                    nodes: new vis.DataSet(dataObj.nodes),
                    edges: new vis.DataSet(dataObj.edges)
                }};

                const options = {{
                    autoResize: true,
                    nodes: {{
                        shape: 'dot',
                        font: {{ size: 15, color: '#1f2937' }},
                        borderWidth: 1,
                        borderWidthSelected: 2
                    }},
                    edges: {{
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
                        font: {{ align: 'middle', size: 12, color: '#111827', strokeWidth: 0 }},
                        color: {{ color: '#6b7280', highlight: '#111827' }},
                        smooth: {{ enabled: true, type: 'dynamic' }}
                    }},
                    interaction: {{
                        dragNodes: true,
                        dragView: true,
                        zoomView: true,
                        hover: true,
                        tooltipDelay: 150
                    }},
                    physics: {{
                        enabled: true,
                        stabilization: {{ iterations: 150 }},
                        barnesHut: {{ gravitationalConstant: -2500, springLength: 120, damping: 0.2 }}
                    }}
                }};

                new vis.Network(container, data, options);
            }})();
        </script>
        """

        components.html(graph_html, height=540, scrolling=False)


def render_highlighted_text(text: str, entities: List[Entity]) -> str:
    html_parts: List[str] = []
    cursor = 0

    for entity in entities:
        start, end = int(entity["start"]), int(entity["end"])
        label = str(entity["label"])

        if cursor < start:
            html_parts.append(html.escape(text[cursor:start]))

        chunk = html.escape(text[start:end])
        color = COLOR_MAP.get(label, "#eeeeee")
        label_text = LABEL_NAME_MAP.get(label, label)

        html_parts.append(
            (
                f"<span class='entity' style='background:{color}'>"
                f"{chunk}<small>{label_text}</small></span>"
            )
        )
        cursor = end

    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))

    return "".join(html_parts).replace("\n", "<br>")


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[\u4e00-\u9fff]+|[^\w\s]"
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(pattern, text)]


def to_bio_sequence(text: str, entities: List[Entity]) -> List[Tuple[str, str]]:
    tokens = tokenize_with_spans(text)
    sequence: List[Tuple[str, str]] = []

    for token, t_start, t_end in tokens:
        tag = "O"

        for entity in entities:
            e_start, e_end = int(entity["start"]), int(entity["end"])
            label = str(entity["label"])

            if t_start >= e_start and t_end <= e_end:
                tag = f"B-{label}" if t_start == e_start else f"I-{label}"
                break

        sequence.append((token, tag))

    return sequence


def main() -> None:
    st.markdown(
        """
        <style>
        .entity {
            padding: 2px 6px;
            border-radius: 6px;
            margin: 0 2px;
            display: inline-block;
            line-height: 1.6;
        }
        .entity small {
            margin-left: 6px;
            opacity: 0.8;
            font-size: 0.72em;
            letter-spacing: 0.2px;
        }
        .result-box {
            border: 1px solid #d9d9d9;
            border-radius: 10px;
            padding: 14px;
            min-height: 150px;
            background: #fffdf7;
            font-size: 1rem;
            line-height: 1.8;
            white-space: normal;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("信息抽取与知识图谱构建系统")
    st.caption("Week 8 随堂 Vibe 实验")

    default_text = (
        "Steve Jobs founded Apple in California.\n"
        "马云创立了阿里巴巴，总部位于杭州。"
    )

    text = st.text_area("输入文本", value=default_text, height=200, placeholder="请输入中英文语料...")

    col1, col2 = st.columns([1, 2])
    with col1:
        show_bio = st.checkbox("查看底层标注（BIO）", value=False)
    with col2:
        run = st.button("开始识别", type="primary", use_container_width=True)

    if not run:
        st.info("点击“开始识别”后查看实体高亮或 BIO 标注结果。")
        return

    if not text.strip():
        st.warning("请先输入文本内容。")
        return

    result = extract_info(text)
    entities = result["entities"]
    relations = result["relations"]
    graph_data = convert_to_graph_data(entities, relations)

    st.subheader("识别结果")
    if show_bio:
        bio_seq = to_bio_sequence(text, entities)
        st.code("\n".join(f"{tok}\t{tag}" for tok, tag in bio_seq), language="text")
    else:
        highlighted = render_highlighted_text(text, entities)
        st.markdown(f"<div class='result-box'>{highlighted}</div>", unsafe_allow_html=True)

    st.subheader("实体列表")
    if entities:
        table_rows = [
            {
                "Text": str(item["text"]),
                "Type": LABEL_NAME_MAP.get(str(item["label"]), str(item["label"])),
                "Start": int(item["start"]),
                "End": int(item["end"]),
            }
            for item in entities
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.write("未识别到实体（可先尝试示例文本中的实体词）。")

    st.subheader("关系抽取结果（Subject - Predicate - Object）")
    if relations:
        relation_rows = [
            {
                "Subject": item["source"],
                "Predicate": item["relation"],
                "Object": item["target"],
            }
            for item in relations
        ]
        st.dataframe(relation_rows, use_container_width=True, hide_index=True)
    else:
        st.write("未抽取到实体关系（可尝试包含“创立/位于/founded/located in”等关系词的文本）。")

    st.subheader("知识图谱可视化")
    if graph_data["nodes"]:
        render_knowledge_graph(graph_data["nodes"], graph_data["edges"])
    else:
        st.write("暂无可视化节点，请先输入可识别实体的文本。")

    st.caption("说明：当前实现为平铺实体标注与规则关系抽取，嵌套实体与复杂关系可在后续模块扩展。")


main()
