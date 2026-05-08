import re
from itertools import combinations
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import gensim.downloader as api
from gensim.models import FastText
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    st.set_page_config(page_title="语义分析综合测试平台", layout="wide")
    main()


DEFAULT_TEXT = """
Natural language processing is a field of artificial intelligence that studies how computers
and humans communicate through language. Researchers build representations of words, sentences,
and documents so that machines can analyze meaning, sentiment, and context. Traditional methods
often begin with counting word frequencies and weighting terms by inverse document frequency.
These approaches are interpretable and efficient for many applications. Modern methods use
distributed representations where words are mapped into dense vectors learned from data.
In these vector spaces, semantically related words tend to appear close to one another.
For example, words about medicine may share similar contexts, while words about finance form
another cluster. Dimensionality reduction techniques can project high-dimensional vectors into
two dimensions for visualization. Such visualizations help learners understand latent semantic
structure and compare model behavior. In teaching and experimentation, interactive tools are
useful because students can modify text and immediately observe numerical and geometric changes.
By combining statistical features with low-rank decomposition, we can demonstrate both classic
information retrieval ideas and the intuition behind semantic spaces.
""".strip()


def ensure_nltk_punkt() -> None:
    """Ensure sentence tokenizer resources are available."""
    need_download = False
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        need_download = True
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        need_download = True

    if need_download:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> List[str]:
    """Split raw text into sentence-level documents and filter noise."""
    try:
        ensure_nltk_punkt()
        raw_sentences = sent_tokenize(text)
    except Exception:
        # Fallback: when NLTK resources cannot be downloaded in cloud runtime.
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in raw_sentences:
        s = sentence.strip()
        if len(s) < 3:
            continue
        # Keep sentences that include at least one alphabetic token.
        if re.search(r"[A-Za-z]", s):
            cleaned.append(s)
    return cleaned


def build_tfidf(sentences: List[str]):
    """Build TF-IDF matrix and return vectorizer + dense dataframe."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    terms = vectorizer.get_feature_names_out()
    df = pd.DataFrame(matrix.toarray(), columns=terms)
    return vectorizer, matrix, df


def top_keywords(tfidf_matrix, terms: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Extract top keywords by global TF-IDF weight."""
    weights = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    if weights.size == 0:
        return pd.DataFrame(columns=["keyword", "weight"])
    top_idx = np.argsort(weights)[::-1][:top_n]
    return pd.DataFrame({"keyword": terms[top_idx], "weight": weights[top_idx]})


def lsa_term_projection(tfidf_matrix, terms: np.ndarray) -> pd.DataFrame:
    """Project vocabulary terms into 2D space using LSA (TruncatedSVD)."""
    # tfidf_matrix: (n_docs, n_terms). We transpose to get term vectors.
    term_doc = tfidf_matrix.T
    min_dim = min(term_doc.shape)
    if min_dim < 2:
        raise ValueError("词汇或句子数量不足，无法降到 2 维。")

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(term_doc)

    return pd.DataFrame({
        "term": terms,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })


def build_sentence_cooccurrence(sentences: List[str]):
    """Build sentence-level term co-occurrence matrix: X^T X."""
    vectorizer = CountVectorizer(stop_words="english", binary=True)
    x = vectorizer.fit_transform(sentences)
    terms = vectorizer.get_feature_names_out()

    if len(terms) < 2:
        raise ValueError("有效词汇数量不足，无法构建共现矩阵。")

    cooc = (x.T @ x).astype(np.float64)
    cooc_dense = cooc.toarray()
    np.fill_diagonal(cooc_dense, 0.0)
    cooc_df = pd.DataFrame(cooc_dense, index=terms, columns=terms)
    return terms, cooc_dense, cooc_df


def lsa_projection_from_square_matrix(matrix: np.ndarray, terms: np.ndarray) -> pd.DataFrame:
    """Apply TruncatedSVD on a square term-term matrix and return 2D coordinates."""
    min_dim = min(matrix.shape)
    if min_dim < 2:
        raise ValueError("矩阵维度不足，无法降维到 2。")

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(matrix)
    return pd.DataFrame({"term": terms, "x": coords[:, 0], "y": coords[:, 1]})


def select_terms_for_labels(terms: np.ndarray, weights: np.ndarray, max_labels: int) -> set:
    """Pick a subset of high-importance terms to label, reducing text overlap."""
    if len(terms) == 0 or max_labels <= 0:
        return set()
    max_labels = min(max_labels, len(terms))
    idx = np.argsort(weights)[::-1][:max_labels]
    return set(terms[idx])


def add_non_overlapping_labels(
    ax,
    df: pd.DataFrame,
    label_terms: set,
    x_col: str = "x",
    y_col: str = "y",
    term_col: str = "term",
    min_gap_ratio: float = 0.06,
) -> int:
    """Add labels greedily, skipping labels that are too close to already-labeled points."""
    if df.empty or not label_terms:
        return 0

    x_span = max(float(df[x_col].max() - df[x_col].min()), 1e-6)
    y_span = max(float(df[y_col].max() - df[y_col].min()), 1e-6)
    used_points = []
    added = 0

    for _, row in df.iterrows():
        term = row[term_col]
        if term not in label_terms:
            continue

        x = float(row[x_col])
        y = float(row[y_col])
        too_close = False
        for px, py in used_points:
            dx = abs(x - px) / x_span
            dy = abs(y - py) / y_span
            if dx < min_gap_ratio and dy < min_gap_ratio:
                too_close = True
                break

        if too_close:
            continue

        ax.annotate(
            term,
            xy=(x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )
        used_points.append((x, y))
        added += 1

    return added


def compute_pair_stats(terms: np.ndarray, cooc_dense: np.ndarray, lsa_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise co-occurrence and 2D distance for observation tasks."""
    coord_map = {row.term: (row.x, row.y) for row in lsa_df.itertuples(index=False)}
    rows = []
    for i, j in combinations(range(len(terms)), 2):
        term_i = terms[i]
        term_j = terms[j]
        xi, yi = coord_map[term_i]
        xj, yj = coord_map[term_j]
        distance = float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
        rows.append(
            {
                "word_1": term_i,
                "word_2": term_j,
                "cooccurrence": float(cooc_dense[i, j]),
                "lsa_distance": distance,
            }
        )
    pair_df = pd.DataFrame(rows)
    return pair_df.sort_values(["cooccurrence", "lsa_distance"], ascending=[False, True])


def tokenize_for_w2v(text: str) -> List[List[str]]:
    """Convert raw text into sentence-level token lists for Word2Vec."""
    sentences = split_sentences(text)
    tokenized = []
    for sentence in sentences:
        tokens = re.findall(r"[A-Za-z]+", sentence.lower())
        tokens = [tok for tok in tokens if len(tok) > 1]
        if tokens:
            tokenized.append(tokens)
    return tokenized


def train_word2vec_model(
    tokenized_sentences: List[List[str]],
    sg: int,
    window: int,
    vector_size: int,
    min_count: int,
    epochs: int,
) -> Word2Vec:
    """Train a Word2Vec model with user-selected settings."""
    if len(tokenized_sentences) < 2:
        raise ValueError("可训练句子不足，请提供更长文本。")

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=sg,
        epochs=epochs,
        seed=42,
    )
    return model


def train_fasttext_model(
    tokenized_sentences: List[List[str]],
    window: int,
    vector_size: int,
    min_count: int,
    epochs: int,
) -> FastText:
    """Train a FastText model for subword-aware word vectors."""
    if len(tokenized_sentences) < 2:
        raise ValueError("可训练句子不足，请提供更长文本。")

    model = FastText(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        epochs=epochs,
        seed=42,
    )
    return model


def sentence_to_avg_vector(sentence: str, fasttext_model: FastText) -> np.ndarray:
    """Create a sentence vector by averaging token vectors (Sent2Vec-style baseline)."""
    tokens = re.findall(r"[A-Za-z]+", sentence.lower())
    if not tokens:
        raise ValueError("句子中未检测到有效英文词。")

    vectors = [fasttext_model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


@st.cache_resource(show_spinner=False)
def load_glove_twitter_25():
    """Load lightweight pretrained GloVe embeddings from gensim downloader."""
    return api.load("glove-twitter-25")


def render_module_1() -> None:
    st.subheader("模块 1：传统统计模型 (TF-IDF 与 LSA)")
    st.write("输入英文语料后，系统将按句子切分，计算 TF-IDF，提取关键词，并进行 LSA 二维可视化。")

    input_text = st.text_area(
        "请输入英文文本（建议 500-1000 词）",
        value=DEFAULT_TEXT,
        height=260,
        help="可粘贴维基百科段落、文学片段或课程相关文本。",
    )

    run_btn = st.button("运行模块 1 分析", type="primary")

    if not run_btn:
        return

    try:
        if not input_text or len(input_text.split()) < 30:
            st.warning("文本过短。请至少输入约 30 个英文词，以便稳定分析。")
            return

        sentences = split_sentences(input_text)
        if len(sentences) < 2:
            st.warning("有效句子数量不足。请提供包含多句英文的文本。")
            return

        st.success(f"分句完成：共 {len(sentences)} 句。")
        with st.expander("查看句子级文档集合", expanded=False):
            for i, s in enumerate(sentences, 1):
                st.write(f"{i}. {s}")

        vectorizer, tfidf_matrix, tfidf_df = build_tfidf(sentences)
        terms = vectorizer.get_feature_names_out()

        st.markdown("### TF-IDF 矩阵（句子 × 词汇）")
        st.dataframe(tfidf_df, use_container_width=True, height=260)

        kw_df = top_keywords(tfidf_matrix, terms, top_n=5)
        st.markdown("### Top 5 关键词（按全局 TF-IDF 权重）")
        st.dataframe(kw_df, use_container_width=True)

        label_count = st.slider(
            "图中显示的词标签数量（避免重叠）",
            min_value=5,
            max_value=40,
            value=15,
            step=1,
            key="module1_label_count",
        )

        lsa_df = lsa_term_projection(tfidf_matrix, terms)
        tfidf_weights = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        label_terms_tfidf = select_terms_for_labels(terms, tfidf_weights, label_count)
        st.markdown("### LSA 二维词汇空间可视化")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(lsa_df["x"], lsa_df["y"], alpha=0.75)
        labeled_count_1 = add_non_overlapping_labels(ax, lsa_df, label_terms_tfidf)

        ax.set_title("LSA 2D Projection of Vocabulary Terms")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(f"已显示标签 {labeled_count_1} 个（自动跳过重叠位置）。")

        st.markdown("### 观察任务：共现矩阵分解的 LSA")
        st.write("下方使用句子级共现矩阵 (X^T X) 进行分解，用于检验高同现词在二维空间是否更接近。")

        cooc_terms, cooc_dense, cooc_df = build_sentence_cooccurrence(sentences)
        with st.expander("查看共现矩阵（词汇 × 词汇）", expanded=False):
            st.dataframe(cooc_df, use_container_width=True, height=300)

        cooc_lsa_df = lsa_projection_from_square_matrix(cooc_dense, cooc_terms)
        pair_df = compute_pair_stats(cooc_terms, cooc_dense, cooc_lsa_df)
        cooc_weights = cooc_dense.sum(axis=1)
        label_terms_cooc = select_terms_for_labels(cooc_terms, cooc_weights, label_count)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(cooc_lsa_df["x"], cooc_lsa_df["y"], alpha=0.75, color="#2A6F97")
        labeled_count_2 = add_non_overlapping_labels(ax2, cooc_lsa_df, label_terms_cooc)
        ax2.set_title("Co-occurrence LSA 2D Projection")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        st.caption(f"共现图已显示标签 {labeled_count_2} 个（自动跳过重叠位置）。")

        st.markdown("#### 高同现词对与 LSA 距离")
        top_pairs = pair_df[pair_df["cooccurrence"] > 0].head(10)
        if top_pairs.empty:
            st.info("未发现明显同现词对。请尝试更长文本。")
        else:
            st.dataframe(top_pairs, use_container_width=True)

        st.markdown("#### 自定义词对检查")
        selectable_terms = sorted(cooc_terms.tolist())
        col1, col2 = st.columns(2)
        with col1:
            word_a = st.selectbox("词 1", selectable_terms, index=0)
        with col2:
            word_b = st.selectbox("词 2", selectable_terms, index=1 if len(selectable_terms) > 1 else 0)

        if word_a == word_b:
            st.warning("请选取两个不同的词。")
        else:
            row = pair_df[
                ((pair_df["word_1"] == word_a) & (pair_df["word_2"] == word_b))
                | ((pair_df["word_1"] == word_b) & (pair_df["word_2"] == word_a))
            ]
            if row.empty:
                st.info("该词对未出现在统计结果中。")
            else:
                cooc_val = float(row.iloc[0]["cooccurrence"])
                dist_val = float(row.iloc[0]["lsa_distance"])
                st.write(f"同现次数: {cooc_val:.0f}")
                st.write(f"LSA 二维距离: {dist_val:.4f}")
                if cooc_val > 0:
                    st.success("该词对存在同现关系。可结合距离观察是否映射到相近坐标。")
                else:
                    st.info("该词对无同现关系，通常其空间距离会更大。")

    except Exception as exc:
        st.error(f"模块 1 运行失败：{exc}")
        st.exception(exc)


def render_module_2() -> None:
    st.subheader("模块 2：Word2Vec 训练与对比 (CBOW vs Skip-Gram)")
    st.write("基于输入语料实时训练 Word2Vec，并输出目标词的 Top-5 余弦相似词。")

    input_text = st.text_area(
        "请输入英文文本（与模块 1 可保持一致）",
        value=DEFAULT_TEXT,
        height=220,
        key="module2_text",
    )

    arch = st.radio(
        "选择训练架构",
        options=["CBOW", "Skip-Gram"],
        horizontal=True,
        key="module2_arch",
    )
    sg = 0 if arch == "CBOW" else 1

    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.slider("window（上下文窗口）", min_value=2, max_value=10, value=5, step=1, key="module2_window")
    with col2:
        vector_size = st.slider("vector_size（向量维度）", min_value=20, max_value=200, value=80, step=10, key="module2_dim")
    with col3:
        epochs = st.slider("epochs（训练轮数）", min_value=10, max_value=200, value=80, step=10, key="module2_epochs")

    min_count = st.slider("min_count（最小词频）", min_value=1, max_value=5, value=1, step=1, key="module2_min_count")
    query_word = st.text_input("输入待查询单词", value="language", key="module2_query").strip().lower()
    run_btn = st.button("训练并查询相似词", type="primary", key="module2_run")

    if not run_btn:
        return

    try:
        if not input_text or len(input_text.split()) < 30:
            st.warning("文本过短。请至少输入约 30 个英文词。")
            return

        tokenized_sentences = tokenize_for_w2v(input_text)
        if len(tokenized_sentences) < 2:
            st.warning("有效句子不足，无法训练 Word2Vec。")
            return

        model = train_word2vec_model(
            tokenized_sentences=tokenized_sentences,
            sg=sg,
            window=window,
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs,
        )

        vocab = list(model.wv.index_to_key)
        st.success(f"训练完成：{arch}，词表大小 {len(vocab)}，句子数 {len(tokenized_sentences)}。")

        with st.expander("查看词表样例", expanded=False):
            st.write(", ".join(vocab[:80]) if vocab else "词表为空")

        if not query_word:
            st.info("请输入一个查询词以查看相似词结果。")
            return

        if query_word not in model.wv:
            st.warning(f"词 '{query_word}' 不在当前词表中。请尝试词表中的高频词。")
            return

        similar_words = model.wv.most_similar(query_word, topn=5)
        sim_df = pd.DataFrame(similar_words, columns=["word", "cosine_similarity"])
        st.markdown(f"### 与 '{query_word}' 最相似的 5 个词")
        st.dataframe(sim_df, use_container_width=True)

    except Exception as exc:
        st.error(f"模块 2 运行失败：{exc}")
        st.exception(exc)


def render_module_3() -> None:
    st.subheader("模块 3：预训练模型与词类比 (GloVe)")
    st.write("使用预训练 glove-twitter-25，完成词类比 A - B + C 与词义相似度测试。")

    load_btn = st.button("加载 GloVe 模型（glove-twitter-25）", type="primary", key="module3_load")
    if not load_btn and "glove_model_ready" not in st.session_state:
        st.info("点击上方按钮加载模型后再进行词类比与相似度测试。")
        return

    try:
        with st.spinner("正在加载预训练模型，请稍候..."):
            glove_model = load_glove_twitter_25()
        st.session_state["glove_model_ready"] = True
        st.success(f"模型加载完成，词表大小：{len(glove_model.index_to_key)}")

        st.markdown("### 词类比计算器：A - B + C")
        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.text_input("输入 A", value="king", key="module3_a").strip().lower()
        with col2:
            word_b = st.text_input("输入 B", value="man", key="module3_b").strip().lower()
        with col3:
            word_c = st.text_input("输入 C", value="woman", key="module3_c").strip().lower()

        analogy_btn = st.button("计算类比结果", key="module3_analogy_btn")
        if analogy_btn:
            if not all([word_a, word_b, word_c]):
                st.warning("请完整输入 A、B、C 三个单词。")
            else:
                missing = [w for w in [word_a, word_b, word_c] if w not in glove_model]
                if missing:
                    st.warning(f"以下词不在词表中：{', '.join(missing)}")
                else:
                    result_vec = glove_model[word_a] - glove_model[word_b] + glove_model[word_c]
                    neighbors = glove_model.similar_by_vector(result_vec, topn=10)
                    filtered = [(w, s) for w, s in neighbors if w not in {word_a, word_b, word_c}]
                    if not filtered:
                        st.info("未找到合适候选词，请尝试其他输入。")
                    else:
                        best_word, best_score = filtered[0]
                        st.write(f"结果向量最接近的词：{best_word}")
                        st.write(f"相似度分数：{best_score:.4f}")
                        top_df = pd.DataFrame(filtered[:5], columns=["word", "similarity_to_result"])
                        st.dataframe(top_df, use_container_width=True)

        st.markdown("### 两词相似度（Cosine Similarity）")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            sim_word_1 = st.text_input("单词 1", value="happy", key="module3_sim1").strip().lower()
        with s_col2:
            sim_word_2 = st.text_input("单词 2", value="joy", key="module3_sim2").strip().lower()

        sim_btn = st.button("计算两词相似度", key="module3_sim_btn")
        if sim_btn:
            if not sim_word_1 or not sim_word_2:
                st.warning("请完整输入两个单词。")
            else:
                missing = [w for w in [sim_word_1, sim_word_2] if w not in glove_model]
                if missing:
                    st.warning(f"以下词不在词表中：{', '.join(missing)}")
                else:
                    sim_score = float(glove_model.similarity(sim_word_1, sim_word_2))
                    st.write(f"{sim_word_1} 与 {sim_word_2} 的相似度：{sim_score:.4f}")

    except Exception as exc:
        st.error(f"模块 3 运行失败：{exc}")
        st.exception(exc)


def render_module_4() -> None:
    st.subheader("模块 4：子词特征与句向量 (FastText & Sent2Vec)")
    st.write("实时训练 FastText，完成 OOV 对比测试，并用平均池化构建句向量相似度。")

    input_text = st.text_area(
        "请输入英文文本（建议与前面模块一致）",
        value=DEFAULT_TEXT,
        height=220,
        key="module4_text",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.slider("window", min_value=2, max_value=10, value=5, step=1, key="module4_window")
    with col2:
        vector_size = st.slider("vector_size", min_value=20, max_value=200, value=80, step=10, key="module4_dim")
    with col3:
        epochs = st.slider("epochs", min_value=10, max_value=200, value=80, step=10, key="module4_epochs")

    min_count = st.slider("min_count", min_value=1, max_value=5, value=1, step=1, key="module4_min_count")
    train_btn = st.button("训练 FastText 并执行模块 4", type="primary", key="module4_run")

    if train_btn:
        try:
            if not input_text or len(input_text.split()) < 30:
                st.warning("文本过短。请至少输入约 30 个英文词。")
                return

            tokenized_sentences = tokenize_for_w2v(input_text)
            if len(tokenized_sentences) < 2:
                st.warning("有效句子不足，无法训练模型。")
                return

            w2v_model = train_word2vec_model(
                tokenized_sentences=tokenized_sentences,
                sg=1,
                window=window,
                vector_size=vector_size,
                min_count=min_count,
                epochs=epochs,
            )
            fasttext_model = train_fasttext_model(
                tokenized_sentences=tokenized_sentences,
                window=window,
                vector_size=vector_size,
                min_count=min_count,
                epochs=epochs,
            )

            st.session_state["module4_w2v_model"] = w2v_model
            st.session_state["module4_fasttext_model"] = fasttext_model
            st.session_state["module4_trained"] = True
        except Exception as exc:
            st.error(f"模块 4 运行失败：{exc}")
            st.exception(exc)
            return

    if not st.session_state.get("module4_trained"):
        st.info("请先点击“训练 FastText 并执行模块 4”。")
        return

    w2v_model = st.session_state.get("module4_w2v_model")
    fasttext_model = st.session_state.get("module4_fasttext_model")
    if w2v_model is None or fasttext_model is None:
        st.warning("模型状态丢失，请重新训练一次。")
        st.session_state["module4_trained"] = False
        return

    st.success(
        f"训练完成：Word2Vec 词表 {len(w2v_model.wv.index_to_key)}，FastText 词表 {len(fasttext_model.wv.index_to_key)}。"
    )

    st.markdown("### OOV 测试（Word2Vec vs FastText）")
    oov_word = st.text_input("输入一个可能拼写错误的词", value="computeer", key="module4_oov").strip().lower()
    oov_btn = st.button("执行 OOV 对比", key="module4_oov_btn")

    if oov_btn:
        if not oov_word:
            st.warning("请输入待测试词。")
        else:
            try:
                _ = w2v_model.wv[oov_word]
                st.info("Word2Vec：该词在词表中，可提取向量。")
            except KeyError:
                st.warning("Word2Vec：未登录词")

            ft_vec = fasttext_model.wv[oov_word]
            ft_neighbors = fasttext_model.wv.similar_by_vector(ft_vec, topn=5)
            ft_df = pd.DataFrame(ft_neighbors, columns=["word", "similarity"])
            st.write("FastText：可通过子词信息构造向量，最相似词如下：")
            st.dataframe(ft_df, use_container_width=True)

    st.markdown("### Sent2Vec（Average Pooling 简化实现）")
    sent1 = st.text_area(
        "句子 1",
        value="Natural language processing helps computers understand human language in practical applications.",
        key="module4_sent1",
        height=100,
    )
    sent2 = st.text_area(
        "句子 2",
        value="Machines learn semantic patterns from text so they can analyze meaning more effectively.",
        key="module4_sent2",
        height=100,
    )
    sent_btn = st.button("计算句向量相似度", key="module4_sent_btn")

    if sent_btn:
        try:
            vec1 = sentence_to_avg_vector(sent1, fasttext_model)
            vec2 = sentence_to_avg_vector(sent2, fasttext_model)
            sim_score = cosine_similarity(vec1, vec2)
            st.write(f"句向量余弦相似度：{sim_score:.4f}")
        except Exception as exc:
            st.error(f"句向量计算失败：{exc}")


def main():
    st.title("语义分析综合测试平台")
    st.caption("NLP Week 4 Vibe 实验：语义表示与对比分析系统")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "模块 1：TF-IDF 与 LSA",
            "模块 2：Word2Vec/GloVe/FastText",
            "模块 3：预训练 GloVe 与词类比",
            "模块 4：FastText 与 Sent2Vec",
        ]
    )

    with tab1:
        render_module_1()

    with tab2:
        render_module_2()

    with tab3:
        render_module_3()

    with tab4:
        render_module_4()

if __name__ == "__main__":
    st.set_page_config(page_title="语义分析综合测试平台", layout="wide")
    main()
