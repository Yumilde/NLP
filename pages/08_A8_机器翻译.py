import streamlit as st

from shared.adapter import render_a8

st.set_page_config(page_title="A8 机器翻译", layout="wide")
st.title("A8：机器翻译对比与 BLEU 评测")

st.markdown("### 实验任务")
st.write("比较规则直译与神经机器翻译，并用 BLEU 完成自动评测。")

st.markdown("### 可交互演示")
render_a8()

st.markdown("### 结果对比与结论")
st.info("建议展示一个歧义表达案例，突出 NMT 在语义意译上的优势。")
