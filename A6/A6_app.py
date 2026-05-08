import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import nltk
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline


DEFAULT_TEXT = (
    "Natural language processing combines linguistics and machine learning. "
    "Language models estimate how likely a sentence is. "
    "Smoothing helps when some word combinations are unseen in training data."
)

DEFAULT_CHAR_CORPUS = (
    "Two roads diverged in a yellow wood,\n"
    "And sorry I could not travel both\n"
    "And be one traveler, long I stood\n"
    "And looked down one as far as I could."
)


class CharRNNLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, model_type: str = "RNN"):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if model_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits


@st.cache_data(show_spinner=False)
def load_reuters_sample(max_sentences: int = 300) -> str:
    """Load a small Reuters sample as baseline corpus text."""
    try:
        nltk.download("reuters", quiet=True)
        from nltk.corpus import reuters

        sentences = reuters.sents()[:max_sentences]
        joined = [" ".join(sent) for sent in sentences if sent]
        if joined:
            return " ".join(joined)
    except Exception:
        pass
    return DEFAULT_TEXT


def simple_tokenize(text: str) -> List[str]:
    """Regex tokenizer that keeps words and apostrophes."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def build_ngram_model(tokens: Sequence[str], n: int) -> Tuple[Dict[Tuple[str, ...], Counter], Counter, int]:
    """Return context->next_word counts, context totals, and vocabulary size."""
    if n < 2:
        raise ValueError("n must be >= 2")

    start_tokens = ["<s>"] * (n - 1)
    padded = start_tokens + list(tokens)

    ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    context_totals: Counter = Counter()

    for i in range(n - 1, len(padded)):
        context = tuple(padded[i - (n - 1) : i])
        word = padded[i]
        ngram_counts[context][word] += 1
        context_totals[context] += 1

    vocab_size = len(set(tokens))
    return ngram_counts, context_totals, vocab_size


def sentence_step_probs(
    sentence_tokens: Sequence[str],
    n: int,
    ngram_counts: Dict[Tuple[str, ...], Counter],
    context_totals: Counter,
    vocab_size: int,
    use_laplace: bool,
) -> List[Dict[str, object]]:
    """Compute conditional probabilities for each prediction step."""
    start_tokens = ["<s>"] * (n - 1)
    padded = start_tokens + list(sentence_tokens)
    rows: List[Dict[str, object]] = []

    for i in range(n - 1, len(padded)):
        context = tuple(padded[i - (n - 1) : i])
        word = padded[i]

        c_hw = ngram_counts.get(context, Counter()).get(word, 0)
        c_h = context_totals.get(context, 0)

        if use_laplace:
            denom = c_h + vocab_size
            prob = (c_hw + 1) / denom if denom > 0 else 0.0
        else:
            prob = c_hw / c_h if c_h > 0 else 0.0

        rows.append(
            {
                "context": " ".join(context),
                "word": word,
                "count(context,word)": c_hw,
                "count(context)": c_h,
                "P(word|context)": prob,
            }
        )

    return rows


def joint_probability(rows: Sequence[Dict[str, object]]) -> float:
    p = 1.0
    for row in rows:
        p *= float(row["P(word|context)"])
    return p


def format_prob(prob: float) -> str:
    if prob == 0:
        return "0"
    if prob < 1e-6:
        return f"{prob:.3e}"
    return f"{prob:.8f}"


def build_char_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text if c in stoi]


def generate_text(
    model: nn.Module,
    seed: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    gen_len: int = 50,
    temperature: float = 1.0,
) -> str:
    if not seed:
        return ""

    model.eval()
    device = next(model.parameters()).device
    generated = seed

    known_seed = "".join(ch for ch in seed if ch in stoi)
    if not known_seed:
        known_seed = next(iter(stoi.keys()))
        generated = known_seed

    x = torch.tensor([[stoi[ch] for ch in known_seed]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(gen_len):
            logits = model(x)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_char = itos[int(next_idx.item())]
            generated += next_char
            x = torch.cat([x, next_idx], dim=1)

    return generated


@st.cache_resource(show_spinner=False)
def get_bert_fill_mask_pipeline():
    return pipeline("fill-mask", model="bert-base-uncased", device=-1)


@st.cache_resource(show_spinner=False)
def get_gpt2_text_generation_pipeline():
    return pipeline("text-generation", model="gpt2", device=-1)


def render_module_1() -> None:
    st.subheader("模块 1：n-gram 语言模型与加一平滑")
    st.caption("基于 nltk 语料构建统计语言模型，计算句子联合概率，并比较平滑前后差异。")

    source = st.radio(
        "语料来源",
        ["NLTK Reuters 示例语料", "手动输入语料"],
        horizontal=True,
    )

    baseline_text = load_reuters_sample() if source == "NLTK Reuters 示例语料" else DEFAULT_TEXT

    corpus_text = st.text_area(
        "2.1 训练语料文本",
        value=baseline_text,
        height=220,
        help="可直接使用 Reuters 示例，也可以粘贴自己的英文语料。",
    )

    n = st.selectbox("n-gram 阶数", options=[2, 3], index=1)

    use_laplace = st.checkbox("2.4 开启加一平滑（Add-one / Laplace Smoothing）", value=False)

    sentence = st.text_input(
        "2.3 输入待评估句子",
        value="language models help with unseen combinations",
    )

    tokens = simple_tokenize(corpus_text)
    sent_tokens = simple_tokenize(sentence)

    if len(tokens) < n:
        st.error("训练语料过短，无法构建当前 n-gram 模型。请增加语料或降低 n。")
        return

    if not sent_tokens:
        st.warning("请先输入一个英文句子。")
        return

    ngram_counts, context_totals, vocab_size = build_ngram_model(tokens, n)

    rows_no = sentence_step_probs(
        sent_tokens,
        n,
        ngram_counts,
        context_totals,
        vocab_size,
        use_laplace=False,
    )
    rows_yes = sentence_step_probs(
        sent_tokens,
        n,
        ngram_counts,
        context_totals,
        vocab_size,
        use_laplace=True,
    )

    p_no = joint_probability(rows_no)
    p_yes = joint_probability(rows_yes)

    zero_events = [r for r in rows_no if float(r["P(word|context)"]) == 0.0]

    col1, col2, col3 = st.columns(3)
    col1.metric("语料词元数", len(tokens))
    col2.metric("词表大小 V", vocab_size)
    col3.metric("唯一上下文数", len(context_totals))

    st.markdown("### 2.2 n-gram 统计示例")
    top_contexts = sorted(context_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    st.dataframe(
        [
            {
                "context": " ".join(ctx),
                "count(context)": cnt,
                "top next words": ", ".join(
                    f"{w}:{c}" for w, c in ngram_counts[ctx].most_common(3)
                ),
            }
            for ctx, cnt in top_contexts
        ],
        use_container_width=True,
    )

    st.markdown("### 2.3 句子联合概率")
    if use_laplace:
        st.success(f"当前模式：已开启平滑，P(sentence) = {format_prob(p_yes)}")
        st.dataframe(rows_yes, use_container_width=True)
    else:
        st.info(f"当前模式：未平滑，P(sentence) = {format_prob(p_no)}")
        st.dataframe(rows_no, use_container_width=True)

    st.markdown("### 2.5 零概率事件与平滑前后对比")
    if zero_events:
        st.warning("检测到未见过的 n-gram，未平滑联合概率为 0。")
        st.write(f"未平滑 P(sentence) = {format_prob(p_no)}")
        st.write(f"加一平滑 P(sentence) = {format_prob(p_yes)}")
        st.dataframe(
            [
                {
                    "context": row["context"],
                    "word": row["word"],
                    "count(context,word)": row["count(context,word)"],
                    "count(context)": row["count(context)"],
                }
                for row in zero_events
            ],
            use_container_width=True,
        )
    else:
        st.success("当前输入中未出现零概率 n-gram。")
        st.write(f"未平滑 P(sentence) = {format_prob(p_no)}")
        st.write(f"加一平滑 P(sentence) = {format_prob(p_yes)}")


def render_module_2() -> None:
    st.subheader("模块 2：从零训练 RNN 语言模型（字符级）")
    st.caption("使用 PyTorch 在小语料上训练字符级自回归模型，观察 Loss 收敛并进行续写。")

    corpus = st.text_area(
        "2.1 输入短语料",
        value=DEFAULT_CHAR_CORPUS,
        height=180,
        help="建议输入一小段英文诗句、名言或短文。",
    )

    model_type = st.selectbox("模型类型", ["RNN", "LSTM"], index=0)
    hidden_size = st.slider("2.2 Hidden Size", min_value=16, max_value=128, value=64, step=8)
    epochs = st.slider("2.2 Epochs", min_value=10, max_value=200, value=60, step=10)
    learning_rate = st.slider("2.2 Learning Rate", min_value=0.001, max_value=0.1, value=0.02, step=0.001)

    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.stoi = None
        st.session_state.itos = None
        st.session_state.training_meta = {}

    start_train = st.button("2.3 开始训练", type="primary")

    clean_corpus = corpus
    if len(clean_corpus) < 20:
        st.warning("语料太短，建议至少输入 20 个字符。")
        return

    chars, stoi, itos = build_char_vocab(clean_corpus)
    encoded = encode_text(clean_corpus, stoi)
    if len(encoded) < 2:
        st.error("可用字符过少，无法构建训练样本。")
        return

    st.write(f"字符表大小：{len(chars)}，训练序列长度：{len(encoded) - 1}")

    if start_train:
        device = torch.device("cpu")
        model = CharRNNLM(vocab_size=len(chars), hidden_size=hidden_size, model_type=model_type).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        x = torch.tensor([encoded[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([encoded[1:]], dtype=torch.long, device=device)

        chart = st.line_chart([])
        progress = st.progress(0)
        status = st.empty()

        losses: List[float] = []
        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, len(chars)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)
            chart.add_rows({"loss": [loss_value]})
            progress.progress(int(epoch / epochs * 100))
            status.write(f"Epoch {epoch}/{epochs} - Loss: {loss_value:.4f}")

        st.session_state.trained_model = model.cpu()
        st.session_state.stoi = stoi
        st.session_state.itos = itos
        st.session_state.training_meta = {
            "hidden_size": hidden_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "model_type": model_type,
            "final_loss": losses[-1],
        }

        st.success(f"训练完成。最终 Loss: {losses[-1]:.4f}")

    st.markdown("### 2.4 基于 Seed 的文本生成")
    seed = st.text_input("输入起始字符（Seed）", value="And ")
    gen_len = st.slider("生成长度", min_value=20, max_value=120, value=50, step=10)

    if st.button("生成文本"):
        model = st.session_state.trained_model
        stoi_state = st.session_state.stoi
        itos_state = st.session_state.itos

        if model is None or stoi_state is None or itos_state is None:
            st.error("请先完成训练，再进行生成。")
            return

        generated = generate_text(
            model=model,
            seed=seed,
            stoi=stoi_state,
            itos=itos_state,
            gen_len=gen_len,
            temperature=1.0,
        )
        st.text_area("生成结果", value=generated, height=140)


def render_module_3() -> None:
    st.subheader("模块 3：预训练架构对比（Masked LM vs. Causal LM）")
    st.caption("使用 Hugging Face transformers pipeline 对比 BERT 掩码预测与 GPT-2 自回归续写。")

    left, right = st.columns(2)

    with left:
        st.markdown("### 2.1 BERT：Masked Language Modeling")
        mask_input = st.text_input(
            "输入包含 [MASK] 的句子",
            value="The man went to the [MASK] to buy some milk.",
            key="bert_mask_input",
        )

        if st.button("运行 BERT 预测", key="run_bert"):
            if "[MASK]" not in mask_input:
                st.error("请输入包含 [MASK] 的句子。")
            else:
                try:
                    with st.spinner("正在加载 bert-base-uncased 并预测..."):
                        bert_pipe = get_bert_fill_mask_pipeline()
                        preds = bert_pipe(mask_input, top_k=5)

                    st.success("BERT Top-5 候选词")
                    st.dataframe(
                        [
                            {
                                "rank": i + 1,
                                "token": item["token_str"].strip(),
                                "probability": float(item["score"]),
                            }
                            for i, item in enumerate(preds)
                        ],
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"BERT 推理失败：{exc}")

    with right:
        st.markdown("### 2.2 GPT-2：Causal Language Modeling")
        prompt = st.text_area(
            "输入前缀 Prompt",
            value="In a small town, people believed that",
            height=100,
            key="gpt2_prompt",
        )

        if st.button("运行 GPT-2 续写", key="run_gpt2"):
            if not prompt.strip():
                st.error("请输入非空 Prompt。")
            else:
                try:
                    with st.spinner("正在加载 gpt2 并生成..."):
                        gpt2_pipe = get_gpt2_text_generation_pipeline()
                        outputs = gpt2_pipe(
                            prompt,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.9,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1,
                            pad_token_id=50256,
                        )

                    generated_text = outputs[0]["generated_text"]
                    st.success("GPT-2 续写结果（新增约 20 个 token）")
                    st.text_area("生成文本", value=generated_text, height=180)
                except Exception as exc:
                    st.error(f"GPT-2 推理失败：{exc}")

    st.markdown("### 2.3 机制对比说明")
    st.info(
        "BERT 在 [MASK] 位置利用左右文共同预测缺失词；GPT-2 严格按照从左到右方式逐步生成后续 token。"
    )


def render_module_4() -> None:
    st.subheader("模块 4：语言模型评价（Perplexity 困惑度）")
    st.caption("基于 GPT-2 的交叉熵损失计算每个句子的困惑度，PPL = exp(Loss)。")

    default_input = (
        "The weather is nice today and we went for a walk in the park.\n"
        "Colorless green ideas sleep furiously.\n"
        "milk quickly table sky of the because"
    )
    multi_text = st.text_area(
        "2.1 输入测试句子（每行一句）",
        value=default_input,
        height=180,
        help="可输入多行英文句子，系统将逐行计算 Loss 和 PPL。",
    )

    if st.button("计算 PPL", key="calc_ppl"):
        lines = [line.strip() for line in multi_text.splitlines() if line.strip()]
        if not lines:
            st.error("请至少输入一行有效句子。")
            return

        try:
            with st.spinner("正在加载 GPT-2 并计算..."):
                gpt2_pipe = get_gpt2_text_generation_pipeline()
                gpt2_model = gpt2_pipe.model
                gpt2_tokenizer = gpt2_pipe.tokenizer
                gpt2_model.eval()
                model_device = next(gpt2_model.parameters()).device

                rows: List[Dict[str, object]] = []
                for sent in lines:
                    encoded = gpt2_tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
                    input_ids = encoded["input_ids"].to(model_device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(model_device)

                    if input_ids.shape[1] < 2:
                        rows.append(
                            {
                                "sentence": sent,
                                "cross_entropy_loss": None,
                                "ppl": None,
                                "note": "句子过短，无法稳定计算。",
                            }
                        )
                        continue

                    with torch.no_grad():
                        outputs = gpt2_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids,
                        )
                        loss_val = float(outputs.loss.item())
                        ppl_val = math.exp(loss_val)

                    rows.append(
                        {
                            "sentence": sent,
                            "cross_entropy_loss": round(loss_val, 6),
                            "ppl": round(ppl_val, 6),
                            "note": "",
                        }
                    )

            st.markdown("### 2.2-2.4 结果表")
            st.dataframe(rows, use_container_width=True)
        except Exception as exc:
            st.error(f"PPL 计算失败：{exc}")


def main() -> None:
    st.title("构建'语言模型训练与对比分析平台'")
    st.caption("Week 7 随堂 Vibe 实验")

    tabs = st.tabs([
        "模块 1：n-gram & Smoothing",
        "模块 2：Train your own RNN-LM",
        "模块 3：Masked LM vs Causal LM",
        "模块 4：Perplexity 评估",
    ])

    with tabs[0]:
        render_module_1()

    with tabs[1]:
        render_module_2()

    with tabs[2]:
        render_module_3()

    with tabs[3]:
        render_module_4()


main()
