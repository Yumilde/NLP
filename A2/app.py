import io
import re
import spacy
import streamlit as st
from spacy import displacy
import nltk
import os

# 说明：下面的依赖会在运行时按需下载模型
try:
    import benepar
except ImportError:
    benepar = None

try:
    import svgling
except ImportError:
    svgling = None


DEFAULT_TEXT = "The boy saw the man with the telescope."


def load_spacy_model(model_name: str):
    """Load spaCy model with graceful fallback (no hard blocking download)."""
    try:
        return spacy.load(model_name), ""
    except Exception as e:
        return None, str(e)


# Monkey-patch for T5Tokenizer compatibility with benepar
def patch_t5_tokenizer():
    try:
        from transformers import T5Tokenizer
        if not hasattr(T5Tokenizer, "build_inputs_with_special_tokens"):
            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                if token_ids_1 is None:
                    return token_ids_0 + [self.eos_token_id]
                return token_ids_0 + token_ids_1 + [self.eos_token_id]
            T5Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    except ImportError:
        pass

patch_t5_tokenizer()

def try_enable_benepar(nlp):
    """Enable benepar automatically, downloading the model if necessary."""
    if benepar is None:
        return False, "未安装 benepar 依赖。"
    
    model_name = "benepar_en3"
    
    # 尝试直接添加
    try:
        if "benepar" not in nlp.pipe_names:
            nlp.add_pipe("benepar", config={"model": model_name})
        return True, ""
    except Exception:
        # 如果失败，尝试下载模型
        try:
            with st.spinner(f"正在下载成分句法模型 {model_name}... 请稍候"):
                # 设置 NLTK 数据路径到用户目录，避免权限问题
                nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
                if nltk_data_path not in nltk.data.path:
                    nltk.data.path.append(nltk_data_path)
                
                benepar.download(model_name)
                
            if "benepar" not in nlp.pipe_names:
                nlp.add_pipe("benepar", config={"model": model_name})
            return True, ""
        except Exception as exc:
            return False, f"模型下载或加载失败: {str(exc)}"


def build_dependency_svg(nlp, text: str) -> str:
    """生成依存句法的 SVG 字符串。"""
    doc = nlp(text)
    svg = displacy.render(doc, style="dep", jupyter=False, options={"distance": 80})
    return svg


def build_constituency_tree(nlp, text: str):
    """生成成分句法树；优先返回 svgling 图形，否则返回文本树。"""
    doc = nlp(text)
    sent = list(doc.sents)[0]
    tree = sent._.parse_string
    if svgling is not None:
        from nltk import Tree
        nltk_tree = Tree.fromstring(tree)
        return svgling.draw_tree(nltk_tree)
    return tree


def extract_core_arguments(nlp, text: str):
    """从句子中提取核心三元组 (nsubj, ROOT, dobj/pobj)。"""
    doc = nlp(text)
    rows = []
    for token in doc:
        if token.dep_ == "ROOT":
            # 查找主语
            subj = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            # 查找宾语 (dobj) 或介宾 (pobj)
            obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr")]
            # 处理介宾结构
            if not obj:
                for child in token.rights:
                    if child.dep_ == "prep":
                        obj += [w.text for w in child.rights if w.dep_ == "pobj"]

            rows.append(
                {
                    "主体 (Subject)": ", ".join(subj) if subj else "[缺失]",
                    "动作 (Verb)": token.text,
                    "客体 (Object)": ", ".join(obj) if obj else "[缺失]",
                }
            )
    return rows


def main() -> None:
    st.title("句法双引擎透视仪与'歧义侦探'")
    st.caption("Week 3 随堂 Vibe 实验")

    text = st.text_area("输入英文句子（仅限单句）", value=DEFAULT_TEXT, height=100)

    # 载入基础模型
    nlp, err = load_spacy_model("en_core_web_sm")
    if nlp is None:
        st.error(f"基础模型加载失败：{err}")
        st.info("可稍后重试；若在本地运行，请执行：python -m spacy download en_core_web_sm")
        return

    # 自动尝试启用成分句法
    benepar_ready, benepar_err = try_enable_benepar(nlp)

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
            st.warning("成分句法模型未就绪（benepar_en3）。当前环境仅展示依存句法与核心论元。")
            if benepar_err:
                st.caption(f"错误详情：{benepar_err}")
        else:
            try:
                tree = build_constituency_tree(nlp, text)
                if svgling is not None and hasattr(tree, "_repr_svg_"):
                    svg = tree._repr_svg_()
                    st.components.v1.html(svg, height=500, scrolling=True)
                else:
                    st.code(tree)
            except Exception as exc:
                st.error(f"成分句法渲染失败：{exc}")

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
else:
    main()
