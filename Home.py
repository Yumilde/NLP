import streamlit as st

st.set_page_config(page_title="NLP Vibe Coding Showcase", page_icon="📚", layout="wide")

st.title("《自然语言处理》Vibe Coding 作业总展示")
st.caption("统一入口：A1-A9 | Streamlit 多页面应用")

st.markdown(
    """
本应用用于课程展示，包含 9 次 Vibe Coding 作业：
- A1：中文词法分析（已从 Flask 迁移为 Streamlit）
- A2：句法分析与核心论元提取
- A3：语义表示与相似度
- A4：词义消歧与 SRL
- A5：篇章分析
- A6：语言模型训练与评估
- A7：信息抽取与知识图谱
- A8：机器翻译与 BLEU
- A9：情感分析与意见挖掘

请在左侧 `Pages` 导航中选择对应作业页面。
"""
)

st.info("首次打开含预训练模型的页面可能需要较长加载时间（取决于网络与缓存）。")
