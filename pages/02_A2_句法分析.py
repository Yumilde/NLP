import streamlit as st

from shared.adapter import render_a2

st.set_page_config(page_title="A2 句法分析", layout="wide")
st.title("A2：句法分析（依存关系与成分结构）")

st.markdown("### 实验任务")
st.write("对输入句子进行依存句法和成分句法分析，并提取核心论元。")

st.markdown("### 可交互演示")
render_a2()

st.markdown("### 结果对比与结论")
st.info("建议用同一句子比较依存图与成分树：前者突出词间关系，后者突出层级结构。")
