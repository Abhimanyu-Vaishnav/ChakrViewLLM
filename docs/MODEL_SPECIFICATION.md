# ChakrGenesis Model Specification

## 1. Architecture Overview

ChakrGenesis is a compact, custom-engineered decoder-only autoregressive transformer trained from scratch. It is designed specifically for efficient bilingual (English & Hindi) conversational generation under extreme computational and parameter constraints.

```
       Input Token IDs [B, T]
                 │
                 ▼
      Embedding Layer (Vocab: 4,096, d=256)
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Transformer       Rotary Positional
   Block (x6)        Embeddings (RoPE)
        │                 │
        ├─────────────────┘
        ▼
   RMSNorm (eps=1e-5)
        │
        ▼
   Linear Head [Tied Weights with Embedding]
        │
        ▼
     Logits [B, T, 4096]
```

---

## 2. Structural & Hyperparameter Constants

| Hyperparameter | Symbol | Value | Architectural Justification |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size** | $V$ | `4,096` | Optimized BPE vocabulary covering ASCII and Devanagari Unicode code points without excessive sparse embedding overhead. |
| **Context Window** | $T_{\max}$ | `256` tokens | Sized for conversational turns and low-memory training on consumer hardware. |
| **Model Dimension** | $d_{\text{model}}$ | `256` | Latent channel dimension balancing representational expressiveness and small-scale parameter budgets. |
| **Transformer Layers** | $L$ | `6` | Deep enough for syntactic and semantic induction; shallow enough to prevent vanishing gradients and overfitting on compact corpora. |
| **Query Attention Heads** | $n_{q}$ | `8` | Head dimension $d_{h} = d_{\text{model}} / n_{q} = 32$. Enables multi-aspect attention routing. |
| **Key/Value Attention Heads**| $n_{kv}$ | `2` | **Grouped-Query Attention (GQA)** with query-to-KV ratio of 4:1. Reduces KV cache memory footprints during autoregressive generation by 75%. |
| **Head Dimension** | $d_{h}$ | `32` | Canonical head projection dimension. |
| **Feed-Forward Dimension** | $d_{\text{ff}}$ | `682` | Computed via $\lfloor \frac{8}{3} d_{\text{model}} \rfloor = 682$. Sized specifically for SwiGLU gated activations. |
| **Activation Function** | $\sigma$ | `SwiGLU` | Swish-Gated Linear Unit provides superior gradient flow and representation compared to standard ReLU or GeLU. |
| **Normalization** | — | `RMSNorm` | Root Mean Square Layer Normalization with $\epsilon = 1 \times 10^{-5}$, applied pre-attention and pre-FFN with no learned biases. |
| **Positional Encoding** | — | `RoPE` | Rotary Position Embeddings applied directly to Queries and Keys; base frequency $\theta = 10,000.0$. |
| **Weight Tying** | — | `Enabled` | Output projection head ($W_{\text{out}} \in \mathbb{R}^{V \times d_{\text{model}}}$) shares memory and weight pointers directly with token embeddings ($W_e \in \mathbb{R}^{V \times d_{\text{model}}}$). |
| **Bias Terms** | — | `None` | All linear layers and normalization layers are strictly bias-free, improving training stability and reducing parameter count. |

---

## 3. Mathematical Formulations

### 3.1 Rotary Position Embedding (RoPE)
Given a token at position $m \in [0, T-1]$ and a 2D component of a query or key vector $[x_0, x_1]^T$:
$$\mathbf{R}_{\Theta, m}^d = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}$$
where $\theta_i = 10000^{-2(i-1)/d_h}$ for $i \in \{1, \dots, d_h/2\}$. RoPE allows the inner product $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle$ to depend purely on the relative distance $m - n$.

### 3.2 Grouped-Query Attention (GQA)
With $n_q = 8$ and $n_{kv} = 2$, each KV head is shared across 4 query heads ($g = 4$):
$$\text{Attention}(\mathbf{Q}_g, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}_g \mathbf{K}^T}{\sqrt{d_h}} + \mathbf{M}\right) \mathbf{V}$$
where $\mathbf{M}_{ij} = -\infty$ for $j > i$ (causal autoregressive mask).

### 3.3 SwiGLU Feed-Forward Network
The feed-forward block utilizes gated bilinear units:
$$\text{SwiGLU}(\mathbf{x}) = \left(\text{swish}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}})\right) \mathbf{W}_{\text{down}}$$
where:
- $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $\text{swish}(z) = z \cdot \text{sigmoid}(z)$

### 3.4 Pre-Layer Normalization (RMSNorm)
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$
where $\boldsymbol{\gamma} \in \mathbb{R}^{d}$ is a learnable scaling parameter and $\epsilon = 1 \times 10^{-5}$.

---

## 4. Exact Parameter Accounting

The model comprises **5,177,600 total parameters** (5.18M), of which **4,129,024 are unique non-tied parameters**.

| Component | Tensor Name / Dimension | Parameters per Layer | Layers | Total Parameters | Non-Tied Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | `tok_embeddings.weight` $[4096, 256]$ | — | — | 1,048,576 | 1,048,576 |
| **Attention Query** | `attn.wq.weight` $[256, 256]$ | 65,536 | 6 | 393,216 | 393,216 |
| **Attention Key** | `attn.wk.weight` $[64, 256]$ | 16,384 | 6 | 98,304 | 98,304 |
| **Attention Value** | `attn.wv.weight` $[64, 256]$ | 16,384 | 6 | 98,304 | 98,304 |
| **Attention Output** | `attn.wo.weight` $[256, 256]$ | 65,536 | 6 | 393,216 | 393,216 |
| **Attn Norm** | `attn_norm.weight` $[256]$ | 256 | 6 | 1,536 | 1,536 |
| **FFN Gate** | `ffn.w_gate.weight` $[682, 256]$ | 174,592 | 6 | 1,047,552 | 1,047,552 |
| **FFN Up** | `ffn.w_up.weight` $[682, 256]$ | 174,592 | 6 | 1,047,552 | 1,047,552 |
| **FFN Down** | `ffn.w_down.weight` $[256, 682]$ | 174,592 | 6 | 1,047,552 | 1,047,552 |
| **FFN Norm** | `ffn_norm.weight` $[256]$ | 256 | 6 | 1,536 | 1,536 |
| **Final Norm** | `norm.weight` $[256]$ | — | — | 256 | 256 |
| **Output Head** | `output.weight` $[4096, 256]$ | — | — | 1,048,576 | 0 *(Tied to Embedding)* |
| **Total** | — | — | — | **5,177,600** | **4,129,024** |

---

## 5. Inference & Generation Configuration

The recommended generation parameters for conversational balance:

```json
{
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.85,
  "repetition_penalty": 1.15,
  "no_repeat_ngram_size": 3,
  "eos_token_id": 2,
  "pad_token_id": 0
}
```

- **Repetition Penalty (1.15):** Suppresses unigram cycling without damaging grammatical markers.
- **n-gram Blocking (3-gram):** Prevents periodic phrase cycling (e.g., *"I am... I am..."*).
- **Nucleus Sampling ($p=0.85$):** Eliminates low-probability noise tokens from the tail of the distribution.
