import streamlit as st

from shared.adapter import render_a5

st.set_page_config(page_title="A5 篇章分析", layout="wide")
st.title("A5：篇章分析综合平台")

st.markdown("### 实验任务")
st.write("完成 EDU 切分、显式关系抽取与指代消解可视化。")

st.markdown("### 可交互演示")
render_a5()

st.markdown("### 结果对比与结论")
st.info("可强调：从句法层面进入篇章层面后，任务从词句理解升级为跨句关系建模。")
