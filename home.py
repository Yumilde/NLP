import streamlit as st

st.title("NLP Vibe Coding Portfolio 🚀")
st.markdown("""
### 课程：自然语言处理 (2025-2026-2)
**学生：[您的名字]**

欢迎来到我的 NLP Vibe Coding 作业展示平台！本项目整合了本学期 9 次核心实验，涵盖了从底层词法分析到高层篇章理解与应用的完整流程。

---

#### 🛠️ 核心模块概览

1.  **词法与句法分析 (A1-A2)**:
    - 实现多种中文分词算法（FMM/BMM/BiMM）对比。
    - 结合 spaCy 与 Benepar 实现深度依存句法与成分句法分析。
2.  **语义表示与角色标注 (A3-A4)**:
    - 探索 TF-IDF, LSA, Word2Vec, GloVe, FastText 等词向量技术。
    - 实现词义消歧 (WSD) 与语义角色标注 (SRL)。
3.  **篇章分析与语言模型 (A5-A6)**:
    - 实现 EDU 话语单元切分与显式连词检测。
    - 构建 n-gram 语言模型与简单 Char-RNN 模型。
4.  **知识抽取与综合应用 (A7-A9)**:
    - 自动提取实体关系并构建知识图谱可视化。
    - 实现神经机器翻译 (NMT) 与 BLEU 质量评测。
    - 针对电商评论进行细粒度情感分析与意见挖掘。

---

#### 💡 如何使用？
请通过左侧导航栏切换不同的作业页面。每个页面均提供实时交互功能，您可以输入自己的文本进行测试。

---
*Powered by Streamlit & Hugging Face Transformers*
""", unsafe_allow_html=True)
