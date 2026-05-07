import streamlit as st

# 统一设置页面标题和图标（必须是第一个 Streamlit 命令）
st.set_page_config(page_title="NLP Vibe Coding Portfolio", layout="wide")

# 定义各个作业页面
pages = {
    "项目概览": [
        st.Page("home.py", title="Vibe Coding Portfolio", icon="🏠"),
    ],
    "词法与句法": [
        st.Page("A1/streamlit_app.py", title="A1: 中文词法分析", icon="🔍"),
        st.Page("A2/app.py", title="A2: 句法透视仪", icon="🌲"),
    ],
    "语义与角色": [
        st.Page("A3/A3_app.py", title="A3: 语义分析综合", icon="🧠"),
        st.Page("A4/A4_app.py", title="A4: WSD & SRL", icon="🏷️"),
    ],
    "篇章与逻辑": [
        st.Page("A5/A5_app.py", title="A5: 篇章分析", icon="📝"),
        st.Page("A6/A6_app.py", title="A6: 语言模型", icon="🤖"),
    ],
    "知识与应用": [
        st.Page("A7/A7_app.py", title="A7: 知识抽取", icon="🕸️"),
        st.Page("A8/A8_app.py", title="A8: 机器翻译评测", icon="🌐"),
        st.Page("A9/A9_app.py", title="A9: 情感分析意见挖掘", icon="📊"),
    ]
}

# 运行导航
pg = st.navigation(pages)
pg.run()
