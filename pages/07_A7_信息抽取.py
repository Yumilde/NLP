import streamlit as st

from shared.adapter import render_a7

st.set_page_config(page_title="A7 信息抽取", layout="wide")
st.title("A7：命名实体识别、关系抽取与知识图谱")

st.markdown("### 实验任务")
st.write("完成实体识别（BIO）、关系抽取与图谱可视化联动展示。")

st.markdown("### 可交互演示")
render_a7()

st.markdown("### 结果对比与结论")
st.info("建议展示‘文本 -> 实体 -> 关系 -> 图谱’四步链路，突出结构化价值。")
