import streamlit as st

st.set_page_config(page_title="NLP Vibe Coding Showcase", page_icon="📚", layout="wide")

st.title("自然语言处理课程 Vibe Coding 作业总览")
st.caption("A1-A9 统一展示入口 | 分享主线：结果对比")

st.markdown("""
### 使用方式
1. 左侧边栏选择 `A1` 到 `A9` 页面。
2. 每页按 `实验任务 -> 可交互演示 -> 结果对比与结论` 顺序展示。
3. 若在线环境模型加载较慢，优先使用页面中的默认输入快速复现结果。
""")

st.markdown("### 作业目录速览")
items = [
    ("A1", "中文词法分析：规范化、分词对比、词性标注"),
    ("A2", "句法分析：依存关系与成分结构"),
    ("A3", "语义表示：TF-IDF/LSA/词向量"),
    ("A4", "语义任务：WSD 与 SRL"),
    ("A5", "篇章分析：EDU、显式关系、指代消解"),
    ("A6", "语言模型：统计LM、RNN、预训练模型、PPL"),
    ("A7", "信息抽取：NER、关系抽取、知识图谱"),
    ("A8", "机器翻译：NMT、规则对比、BLEU"),
    ("A9", "情感分析与意见挖掘可视化"),
]

for aid, desc in items:
    st.markdown(f"- **{aid}**: {desc}")

st.info("课堂建议节奏（10-15 分钟）：A1 快速开场 -> A2-A4 结构/语义 -> A5-A7 高阶任务 -> A8-A9 应用闭环。")
