import re
import importlib
import zipfile
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import torch
from nltk.wsd import lesk
from transformers import AutoModel, AutoTokenizer


@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


@st.cache_resource
def ensure_nltk_data():
    data_dir = Path.home() / "nltk_data"
    corpora_dir = data_dir / "corpora"
    data_dir.mkdir(parents=True, exist_ok=True)
    corpora_dir.mkdir(parents=True, exist_ok=True)

    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))

    for pkg in ["wordnet", "omw-1.4"]:
        nltk.download(pkg, download_dir=str(data_dir), quiet=True)
        zip_path = corpora_dir / f"{pkg}.zip"
        target_dir = corpora_dir / pkg
        if zip_path.exists() and not target_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(corpora_dir)

    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/omw-1.4")


@st.cache_resource
def load_spacy_model():
    spacy_module = importlib.import_module("spacy")
    try:
        return spacy_module.load("en_core_web_sm")
    except Exception:
        try:
            spacy_module.cli.download("en_core_web_sm")
            return spacy_module.load("en_core_web_sm")
        except Exception:
            return None


def find_target_span(sentence: str, target_word: str) -> Optional[tuple[int, int]]:
    pattern = rf"\b{re.escape(target_word.strip())}\b"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if not match:
        return None
    return match.start(), match.end()


def get_target_embedding(sentence: str, target_word: str, tokenizer, model) -> Optional[torch.Tensor]:
    span = find_target_span(sentence, target_word)
    if span is None:
        return None

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.last_hidden_state[0]
    start_char, end_char = span

    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        overlap = not (tok_end <= start_char or tok_start >= end_char)
        if overlap:
            token_indices.append(idx)

    if not token_indices:
        return None

    target_vectors = hidden_states[token_indices, :]
    return target_vectors.mean(dim=0)


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    v1 = vec1.cpu().numpy()
    v2 = vec2.cpu().numpy()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def run_lesk(sentence: str, target_word: str):
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence.lower())
    target = target_word.strip().lower()

    # Prefer noun sense first for classic bank-style ambiguity, then fallback.
    synset = lesk(tokens, target, "n")
    if synset is None:
        synset = lesk(tokens, target)
    return synset


def extract_srl_heuristic(sentence: str, nlp):
    doc = nlp(sentence)

    predicate = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            predicate = token
            break
    if predicate is None:
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token
                break

    a0 = ""
    a1 = ""
    am_loc = []
    am_tmp = []

    if predicate is not None:
        for child in predicate.children:
            if child.dep_ in {"nsubj", "nsubjpass"} and not a0:
                a0 = child.text

            if child.dep_ in {"dobj", "obj", "attr"} and not a1:
                a1 = child.text

            if child.dep_ == "prep":
                prep_phrase = None
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        prep_phrase = f"{child.text} {prep_child.text}"
                        if prep_child.ent_type_ in {"GPE", "LOC", "FAC"}:
                            am_loc.append(prep_phrase)
                        elif prep_child.ent_type_ in {"DATE", "TIME"}:
                            am_tmp.append(prep_phrase)
                        break

                if prep_phrase and prep_phrase not in am_loc and prep_phrase not in am_tmp:
                    if child.text.lower() in {"in", "at", "on", "inside", "near", "from", "to"}:
                        am_loc.append(prep_phrase)

        for token in doc:
            if token.dep_ in {"npadvmod", "advmod"} and token.ent_type_ in {"DATE", "TIME"}:
                if token.text not in am_tmp:
                    am_tmp.append(token.text)
            if token.ent_type_ in {"DATE", "TIME"}:
                span_text = token.text
                if span_text not in am_tmp:
                    am_tmp.append(span_text)

    result = {
        "A0 (Agent)": a0,
        "Predicate": predicate.lemma_ if predicate is not None else "",
        "A1 (Patient)": a1,
        "AM-LOC": "; ".join(dict.fromkeys(am_loc)),
        "AM-TMP": "; ".join(dict.fromkeys(am_tmp)),
    }
    return doc, result


def main():
    st.set_page_config(page_title="WSD & SRL 实验系统", layout="wide")
    st.title("Week 5 Vibe 实验：WSD 与 SRL")

    ensure_nltk_data()
    tokenizer, model = load_bert_model()
    nlp = None
    srl_model_error = ""
    try:
        nlp = load_spacy_model()
    except Exception:
        srl_model_error = (
            "SRL 模块所需的 spaCy 英文模型暂不可用。"
            "系统已尝试自动下载；若仍失败，请稍后重试或本地执行 "
            "`python -m spacy download en_core_web_sm`。"
        )

    tab_wsd, tab_srl = st.tabs(["模块 1：词义消歧 (WSD)", "模块 2：语义角色标注 (SRL)"])

    with tab_wsd:
        st.subheader("WSD 对比测试：Lesk vs BERT 上下文向量")

        sentence_1 = st.text_input(
            "输入第一个包含多义词的句子",
            value="I went to the bank to deposit my money.",
        )
        target_word = st.text_input("输入目标多义词", value="bank")
        sentence_2 = st.text_input(
            "输入第二个句子（用于上下文向量对比）",
            value="I sat by the river bank.",
        )

        if st.button("运行 WSD 对比测试", type="primary"):
            if not sentence_1.strip() or not target_word.strip() or not sentence_2.strip():
                st.error("请完整填写两个句子和目标词。")
            else:
                target_word = target_word.strip()

                st.markdown("### 1) 传统方法：NLTK Lesk")
                synset = run_lesk(sentence_1, target_word)
                if synset is None:
                    st.warning("Lesk 未找到合适词义，请检查目标词是否在句子中或尝试更丰富上下文。")
                else:
                    st.write(f"- Predicted Synset: `{synset.name()}`")
                    st.write(f"- Definition: {synset.definition()}")

                st.markdown("### 2) 上下文向量表示：BERT")
                emb_1 = get_target_embedding(sentence_1, target_word, tokenizer, model)
                emb_2 = get_target_embedding(sentence_2, target_word, tokenizer, model)

                if emb_1 is None:
                    st.error("未能在第一个句子中定位目标词，无法提取 BERT 向量。")
                if emb_2 is None:
                    st.error("未能在第二个句子中定位目标词，无法提取 BERT 向量。")

                if emb_1 is not None and emb_2 is not None:
                    similarity = cosine_similarity(emb_1, emb_2)
                    st.write(f"- 句子1目标词向量维度: `{tuple(emb_1.shape)}`")
                    st.write(f"- 句子2目标词向量维度: `{tuple(emb_2.shape)}`")
                    st.success(f"两个目标词上下文向量的余弦相似度: **{similarity:.4f}**")

    with tab_srl:
        st.subheader("SRL 提取与可视化：基于依存句法的启发式近似")

        if srl_model_error:
            st.warning(srl_model_error)

        srl_sentence = st.text_input(
            "输入英文句子（用于 SRL 分析）",
            value="Apple is manufacturing new smartphones in China this year.",
        )

        if st.button("运行 SRL 分析", type="primary"):
            if nlp is None:
                st.error("当前环境未就绪：spaCy 英文模型未就绪，暂无法运行 SRL。")
            elif not srl_sentence.strip():
                st.error("请输入待分析句子。")
            else:
                doc, srl_result = extract_srl_heuristic(srl_sentence, nlp)

                st.markdown("### 1) 结构化 SRL 输出")
                df = pd.DataFrame([srl_result], columns=["A0 (Agent)", "Predicate", "A1 (Patient)", "AM-LOC", "AM-TMP"])
                st.table(df)

                st.markdown("### 2) 依存关系图（displaCy）")
                spacy_module = importlib.import_module("spacy")
                dep_html = spacy_module.displacy.render(doc, style="dep", page=False)
                st.components.v1.html(dep_html, height=420, scrolling=True)


if __name__ == "__main__":
    main()
