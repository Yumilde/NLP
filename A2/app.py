import io

import spacy
import streamlit as st
from spacy import displacy

# 说明：下面的依赖会在运行时按需下载模型
try:
    import benepar
except ImportError:  # pragma: no cover - 运行时提示安装
    benepar = None

try:
    import svgling
except ImportError:  # pragma: no cover - 运行时回退为文本树
    svgling = None


DEFAULT_TEXT = "The boy saw the man with the telescope."


def load_spacy_model(model_name: str):
    """Load spaCy model with graceful fallback (no hard blocking download)."""
    try:
        return spacy.load(model_name), ""
    except Exception:
        try:
            spacy.cli.download(model_name)
            return spacy.load(model_name), ""
        except Exception as exc:
            return None, str(exc)


def ensure_benepar_model(model_name: str) -> None:
    """若 benepar 模型未安装则自动下载。"""
    if benepar is None:
        return
    # 说明：benepar 在本地没有模型时会抛错，所以先检查并下载
    try:
        benepar.download(model_name)
    except Exception:  # pragma: no cover - 下载失败时由上层提示
        pass


def build_dependency_svg(nlp, text: str) -> str:
    """生成依存句法的 SVG 字符串。"""
    doc = nlp(text)
    # 说明：displacy.render 返回 SVG 字符串
    svg = displacy.render(doc, style="dep", jupyter=False, options={"distance": 80})
    return svg


def build_constituency_tree(nlp, text: str):
    """生成成分句法树；优先返回 svgling 图形，否则返回文本树。"""
    doc = nlp(text)
    # 说明：benepar 会将 constituency 结果挂在扩展属性上
    sent = list(doc.sents)[0]
    tree = sent._.parse_string
    if svgling is not None:
        # 说明：svgling 支持从 nltk.Tree 构建 SVG
        from nltk import Tree

        nltk_tree = Tree.fromstring(tree)
        return svgling.draw_tree(nltk_tree)
    return tree


def extract_core_arguments(nlp, text: str):
    """抽取依存关系中的核心论元（nsubj/dobj/pobj/ROOT）。"""
    doc = nlp(text)
    core_labels = {"nsubj", "dobj", "pobj", "ROOT"}
    rows = []
    for token in doc:
        if token.dep_ in core_labels:
            rows.append(
                {
                    "依存关系": token.dep_,
                    "词": token.text,
                    "词性": token.pos_,
                    "支配词": token.head.text if token.head is not token else "-",
                }
            )
    return rows


def main() -> None:
    st.set_page_config(page_title="句法透视仪", layout="wide")
    st.title("句法双引擎透视仪")

    text = st.text_input("输入一句英文：", value=DEFAULT_TEXT)

    nlp, spacy_err = load_spacy_model("en_core_web_sm")
    if nlp is None:
        st.error(
            "spaCy 英文模型加载失败，当前无法解析句法。"
            f"\n详细错误：{spacy_err}"
        )
        st.info("可稍后重试；若在本地运行，请执行：python -m spacy download en_core_web_sm")
        return

    benepar_ready = False
    benepar_err = ""
    if benepar is not None:
        try:
            ensure_benepar_model("benepar_en3")
            if "benepar" not in nlp.pipe_names:
                nlp.add_pipe("benepar", config={"model": "benepar_en3"})
            benepar_ready = True
        except Exception as exc:
            benepar_err = str(exc)

    dep_tab, const_tab = st.tabs(["依存关系", "成分结构"])

    with dep_tab:
        st.subheader("依存句法 (Dependency Parsing)")
        try:
            dep_svg = build_dependency_svg(nlp, text)
            st.components.v1.html(dep_svg, height=500, scrolling=True)
        except Exception as exc:
            st.error(f"依存句法渲染失败：{exc}")

    with const_tab:
        st.subheader("成分句法 (Constituency Parsing)")
        if not benepar_ready:
            st.warning("成分句法依赖 benepar 模型，当前不可用。可先使用依存句法与核心论元模块。")
            if benepar_err:
                st.caption(f"错误详情：{benepar_err}")
        else:
            try:
                tree = build_constituency_tree(nlp, text)
                if svgling is not None and hasattr(tree, "_repr_svg_"):
                    # 说明：在 Streamlit 中直接显示 SVG
                    svg = tree._repr_svg_()
                    st.components.v1.html(svg, height=500, scrolling=True)
                else:
                    # 说明：若没有 svgling，则展示多级文本树
                    st.code(tree)
            except Exception as exc:
                st.error(
                    "成分句法渲染失败：请确认已安装 benepar 与其模型。"
                    f"\n详细错误：{exc}"
                )

    st.divider()
    st.subheader("核心论元提取器")
    try:
        core_rows = extract_core_arguments(nlp, text)
        if core_rows:
            st.dataframe(core_rows, use_container_width=True)
        else:
            st.info("未检测到 nsubj/dobj/pobj/ROOT 相关依存关系。")
    except Exception as exc:
        st.error(f"核心论元提取失败：{exc}")


if __name__ == "__main__":
    main()
