import streamlit as st

from shared.adapter import render_a3

st.set_page_config(page_title="A3 语义表示", layout="wide")
st.title("A3：语义表示与对比")

st.markdown("### 实验任务")
st.write("完成 TF-IDF/LSA、Word2Vec/FastText/GloVe 等表示方法的对比实验。")

st.markdown("### 可交互演示")
render_a3()

st.markdown("### 结果对比与结论")
st.info("分享时可重点展示关键词权重、词向量近邻和 OOV 行为差异。")
