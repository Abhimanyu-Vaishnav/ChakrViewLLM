# Evaluation Methodology

## Batches 1341–1380: Next-Token Prediction and Causal Language Modeling Benchmarking

### 1. Mathematical Formulation

Let $X = (x_1, x_2, \dots, x_T)$ be a sequence of discrete token IDs $x_t \in \{0, 1, \dots, V-1\}$ where $V = 4096$.

For causal language modeling, the model predicts the conditional probability distribution of the next token given preceding context:
$$P(x_t \mid x_1, \dots, x_{t-1}) = \text{softmax}(z_t)_{x_t}$$
where $z_t \in \mathbb{R}^V$ is the logit vector output at step $t-1$.

#### Input and Target Sequence Construction:
- **Input Tokens:** $X_{\text{in}} = (x_1, x_2, \dots, x_{T-1})$
- **Target Tokens:** $Y = (x_2, x_3, \dots, x_T)$
- **Number of Evaluated Positions:** $N = T - 1$.

#### Token-Level Loss and Perplexity:
The aggregate Cross-Entropy Loss across all $M$ test sequences is:
$$\mathcal{L} = -\frac{1}{\sum_j N_j} \sum_{j=1}^M \sum_{i=1}^{N_j} \log P(y_{j,i} \mid x_{j, \le i})$$

Perplexity is calculated by exponentiating the corpus-level cross-entropy loss:
$$PPL = \exp(\mathcal{L})$$

> **Important Metric Rule:** Perplexity is never computed as the arithmetic mean of per-batch perplexities. The total token-level negative log-likelihood is accumulated and divided by the total number of evaluated target tokens before exponentiation.

---

### 2. Baselines and Comparative Control

| Model / Baseline | Probability Formulation | Theoretical Loss / Properties |
|---|---|---|
| **Uniform Random Baseline** | $P(w) = \frac{1}{V} = \frac{1}{4096}$ | $\mathcal{L} = \ln(4096) \approx 8.3178$ nats<br>$PPL = 4096.00$<br>Top-1: $0.024\%$<br>Top-5: $0.122\%$<br>Top-10: $0.244\%$ |
| **Training Empirical Unigram** | $P(w) = \frac{C_{\text{train}}(w) + 1}{\sum C + V}$ (Laplace smoothed) | Computed from 3,212 training split tokens. Captures marginal unigram distribution without context. |
| **ChakrMicro (Canonical Checkpoint)** | $P(w \mid x_{<t}) = \text{softmax}(\text{ChakrMicro}(x_{<t}))$ | 6-layer Transformer decoder, 3,443,136 unique parameters, evaluated in `model.eval()`. |

---

### 3. Evaluation Environment and Throughput

- **Hardware:** Intel/AMD x86_64 CPU (Single Thread / Local CPU Execution)
- **Evaluation Duration:** 0.39 seconds for 3,456 tokens across 27 held-out documents.
- **Evaluation Throughput:** 8,967.6 tokens / second.
- **Memory Footprint:** < 150 MB RAM.
