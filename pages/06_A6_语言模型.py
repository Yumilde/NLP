import streamlit as st

from shared.adapter import render_a6

st.set_page_config(page_title="A6 语言模型", layout="wide")
st.title("A6：语言模型训练与评测")

st.markdown("### 实验任务")
st.write("覆盖统计语言模型、神经语言模型、预训练模型机制与困惑度评估。")

st.markdown("### 可交互演示")
render_a6()

st.markdown("### 结果对比与结论")
st.info("课堂可用‘同一输入在不同模型的预测差异’串联统计法与神经法演进。")
