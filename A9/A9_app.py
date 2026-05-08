import random
import re
from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline():
    """Load lightweight multilingual sentiment model from Hugging Face."""
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        tokenizer="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )


def normalize_label(raw_label: str) -> str:
    """Normalize model labels to Positive / Negative / Neutral."""
    label = raw_label.strip().lower()
    mapping: Dict[str, str] = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "pos": "Positive",
        "neg": "Negative",
        "neu": "Neutral",
        "label_2": "Positive",
        "label_1": "Neutral",
        "label_0": "Negative",
    }
    return mapping.get(label, raw_label)


def gauge_color(label: str) -> str:
    if label == "Positive":
        return "#22c55e"
    if label == "Negative":
        return "#ef4444"
    return "#f59e0b"


def build_confidence_gauge(score: float, label: str) -> go.Figure:
    """Create a semi-circle gauge chart for confidence score."""
    color = gauge_color(label)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Confidence", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
                "bar": {"color": color, "thickness": 0.55},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 33], "color": "#fee2e2"},
                    {"range": [33, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.8,
                    "value": score * 100,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 60, "b": 10},
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(gauge_shape="angular")
    fig.update_layout(
        annotations=[],
    )
    return fig


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, #e0f2fe 0%, transparent 35%),
                radial-gradient(circle at 90% 20%, #dbeafe 0%, transparent 30%),
                linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        }
        .dashboard-card {
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(4px);
            box-shadow: 0 10px 30px rgba(30, 64, 175, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("电商评论情感分析与意见挖掘平台")
    st.caption("Week 10 Vibe 实验：细粒度情感分析与舆情监测")

    tab1, tab2, tab3 = st.tabs(["模块1：基础情感分类", "模块2：显式 vs 隐式情感", "模块3：可视化聚合（待实现）"])

    with tab1:
        st.subheader("单文本情感极性分析")
        st.write("输入一段中文商品评论，系统将输出情感类别与置信度。")

        input_text = st.text_area(
            "请输入评论文本",
            value="这款耳机音质非常好，续航也很稳定，性价比很高。",
            placeholder="例如：这款耳机音质非常好，续航也很稳定，性价比很高。",
            height=140,
        )

        if st.button("开始分析", type="primary"):
            clean_text = re.sub(r"\s+", " ", input_text).strip()
            if not clean_text:
                st.warning("请输入有效评论文本后再分析。")
                return

            with st.spinner("正在加载模型并进行情感分析..."):
                clf = load_sentiment_pipeline()
                result: List[Dict] = clf(clean_text)

            top = result[0]
            label = normalize_label(str(top.get("label", "Unknown")))
            score = float(top.get("score", 0.0))

            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.metric("情感极性", label)
                st.metric("置信度", f"{score * 100:.2f}%")
            with col2:
                fig = build_confidence_gauge(score, label)
                fig.update_layout(
                    title={
                        "text": f"{label} 情感置信度仪表盘",
                        "x": 0.5,
                        "xanchor": "center",
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            st.success("分析完成。")

    with tab2:
        st.subheader("显式情感 vs. 隐式情感识别")
        st.markdown(
            "**显式情感（Explicit）**：文本中出现了明显褒贬词，"
            "如“太棒了”“很差劲”，模型更容易直接判断情感方向。"
        )
        st.markdown(
            "**隐式情感（Implicit）**：文本不一定出现情感词，"
            "但通过客观事实表达态度，如“手机玩游戏半小时就没电了”。"
        )

        explicit_text = st.text_area(
            "显式情感评价",
            value="这款相机成像效果太棒了，我非常满意。",
            placeholder="例如：这款相机成像效果太棒了，我非常满意。",
            height=120,
            key="explicit_input",
        )
        implicit_text = st.text_area(
            "隐式客观描述",
            value="这台手机玩游戏半小时就没电了。",
            placeholder="例如：这台手机玩游戏半小时就没电了。",
            height=120,
            key="implicit_input",
        )

        if st.button("对比分析", type="primary", key="compare_sentiment"):
            explicit_clean = re.sub(r"\s+", " ", explicit_text).strip()
            implicit_clean = re.sub(r"\s+", " ", implicit_text).strip()

            if not explicit_clean and not implicit_clean:
                st.warning("请至少输入一段文本后再进行对比分析。")
            else:
                with st.spinner("正在进行显式/隐式情感识别..."):
                    clf = load_sentiment_pipeline()

                    explicit_result = None
                    implicit_result = None

                    if explicit_clean:
                        explicit_result = clf(explicit_clean)[0]
                    if implicit_clean:
                        implicit_result = clf(implicit_clean)[0]

                left, right = st.columns(2)

                with left:
                    st.markdown("### 显式情感评价结果")
                    if explicit_result is None:
                        st.info("未输入显式情感文本。")
                    else:
                        exp_label = normalize_label(str(explicit_result.get("label", "Unknown")))
                        exp_score = float(explicit_result.get("score", 0.0))
                        st.metric("情感极性", exp_label)
                        st.metric("置信度", f"{exp_score * 100:.2f}%")

                with right:
                    st.markdown("### 隐式客观描述结果")
                    if implicit_result is None:
                        st.info("未输入隐式客观描述文本。")
                    else:
                        imp_label = normalize_label(str(implicit_result.get("label", "Unknown")))
                        imp_score = float(implicit_result.get("score", 0.0))
                        st.metric("情感极性", imp_label)
                        st.metric("置信度", f"{imp_score * 100:.2f}%")

                st.caption("提示：通常显式情感文本置信度更高；隐式情感更依赖模型对上下文和事实描述的理解。")

    with tab3:
        st.subheader("舆情挖掘与可视化仪表盘")
        st.write("点击按钮生成模拟评论数据，并自动完成批量情感分析与口碑分布展示。")

        if st.button("生成测试舆情数据", type="primary", key="gen_opinion_data"):
            review_pool = [
                "包装精致，物流也很快，体验非常好。",
                "用了两周，运行稳定，性价比很高。",
                "做工一般，但这个价格还能接受。",
                "拍照效果不错，夜景模式表现超出预期。",
                "外观好看，手感也不错。",
                "音质很普通，没有宣传得那么好。",
                "客服响应速度慢，沟通体验一般。",
                "续航表现中规中矩，一天一充。",
                "系统偶尔卡顿，体验不是很流畅。",
                "屏幕有轻微漏光，观感受影响。",
                "用了几天就发热严重，不太满意。",
                "连接蓝牙经常断开，影响日常使用。",
                "收到时有划痕，品控需要提升。",
                "整体表现平平，没有明显亮点。",
                "价格偏高，但功能比较全面。",
            ]
            sample_size = random.randint(10, 15)
            sampled_reviews = random.sample(review_pool, sample_size)

            with st.spinner("正在执行批量情感分析..."):
                clf = load_sentiment_pipeline()
                predictions: List[Dict] = clf(sampled_reviews)

            records = []
            counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for text, pred in zip(sampled_reviews, predictions):
                label = normalize_label(str(pred.get("label", "Unknown")))
                score = float(pred.get("score", 0.0))
                if label not in counts:
                    counts["Neutral"] += 1
                else:
                    counts[label] += 1
                records.append(
                    {
                        "评论内容": text,
                        "情感极性": label,
                        "置信度": f"{score * 100:.2f}%",
                    }
                )

            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Positive", counts["Positive"])
            k2.metric("Negative", counts["Negative"])
            k3.metric("Neutral", counts["Neutral"])
            st.markdown("</div>", unsafe_allow_html=True)

            pie_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Positive", "Negative", "Neutral"],
                        values=[counts["Positive"], counts["Negative"], counts["Neutral"]],
                        hole=0.45,
                        marker={
                            "colors": ["#22c55e", "#ef4444", "#f59e0b"],
                            "line": {"color": "#ffffff", "width": 2},
                        },
                        textinfo="label+percent",
                        insidetextorientation="radial",
                    )
                ]
            )
            pie_fig.update_layout(
                title={"text": "电商评论口碑比例分布", "x": 0.5, "xanchor": "center"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 20, "r": 20, "t": 60, "b": 20},
                height=420,
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            st.dataframe(records, use_container_width=True)


main()
