import streamlit as st

from shared.adapter import render_a4

st.set_page_config(page_title="A4 WSD+SRL", layout="wide")
st.title("A4：词义消歧与语义角色标注")

st.markdown("### 实验任务")
st.write("比较传统 Lesk 与 BERT 上下文表示做 WSD，并完成 SRL 近似抽取。")

st.markdown("### 可交互演示")
render_a4()

st.markdown("### 结果对比与结论")
st.info("建议用多义词 bank 的双语境输入，直观说明上下文向量区分能力。")
