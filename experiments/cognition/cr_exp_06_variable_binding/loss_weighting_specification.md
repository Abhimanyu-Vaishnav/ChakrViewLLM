# Loss Weighting Specification — CR-EXP-06

**Author**: Antigravity Autonomous Research Agent  
**Target Comparison**: Control Arm A (Standard SFT) vs. Treatment Arm B (Relational Target-Weighted SFT)  

---

## 1. Mathematical Formulation

Let input token sequence $X = [x_1, \dots, x_N]$ be partitioned into prompt tokens $X_{\text{prompt}} = [x_1, \dots, x_M]$ and assistant response tokens $X_{\text{response}} = [x_{M+1}, \dots, x_N]$.

For next-token prediction, targets are $Y = [x_2, \dots, x_N]$.

### 1.1 Prompt Masking (Both Arms)
All tokens corresponding to user context and prompts are strictly ignored in the loss:
$$w_t = 0 \quad \text{for } t < M$$
(Implemented by setting `target[t] = -100`).

### 1.2 Control Arm A (Standard Assistant Token Loss)
Every active assistant response token receives equal weighting:
$$w_t = 1.0 \quad \forall t \ge M$$
$$\mathcal{L}_{\text{Arm A}} = \frac{1}{\sum_{t=M}^N 1} \sum_{t=M}^N \text{CrossEntropy}(\hat{y}_t, y_t)$$

### 1.3 Treatment Arm B (Relational Target-Weighted Loss)
Let $\mathcal{S}_{\text{target}}$ denote the set of response token indices that correspond to explicitly marked relational target entity spans (e.g., city, entity value, or updated state).

$$w_t = \begin{cases}
5.0 & \text{if } t \in \mathcal{S}_{\text{target}} \\
1.0 & \text{if } t \ge M \text{ and } t \notin \mathcal{S}_{\text{target}} \\
0.0 & \text{if } t < M \text{ (prompt token)}
\end{cases}$$

$$\mathcal{L}_{\text{Arm B}} = \frac{\sum_{t=M}^N w_t \cdot \text{CrossEntropy}(\hat{y}_t, y_t)}{\sum_{t=M}^N w_t}$$

---

## 2. Token-Span Mapping Algorithm

To prevent span indexing errors between raw Unicode character strings and BPE tokens:
1. Encode prompt: `p_ids = tok.encode(prompt)`
2. Encode response: `r_ids = tok.encode(" " + response)`
3. For each token index $j$ in `r_ids`, compute the reconstructed character slice in `response`.
4. If the token's character boundaries overlap with any span in `target_spans`, assign weight $w_j = 5.0$; otherwise $w_j = 1.0$.
5. If `target_spans` is empty (e.g. conversational preservation dialogue), all response tokens default to weight $1.0$.
6. Zero response loss normalization is guarded: $\sum w_t > 0$.

