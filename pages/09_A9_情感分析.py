import streamlit as st

from shared.adapter import render_a9

st.set_page_config(page_title="A9 情感分析", layout="wide")
st.title("A9：情感分析与意见挖掘")

st.markdown("### 实验任务")
st.write("完成单文本情感分类、显隐式情感对比与批量舆情可视化。")

st.markdown("### 可交互演示")
render_a9()

st.markdown("### 结果对比与结论")
st.info("可用‘单条分析 -> 对比分析 -> 批量仪表盘’展示从模型到业务应用的闭环。")
