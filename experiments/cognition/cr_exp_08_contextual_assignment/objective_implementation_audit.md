# CR-EXP-08: OBJECTIVE IMPLEMENTATION AUDIT
## Synthetic Unit Tests & Loss Gradient Flow Verification

- **Date**: 2026-11-06
- **Architecture**: ChakrMicro v0.1 (3,443,136 parameters)
- **Status**: **ALL OBJECTIVE IMPLEMENTATION TESTS PASSED (GATE 1 CLEARED)**

---

### 1. Objective Formulations Audited

#### Arm A: Standard SFT Control
$$L_A = \frac{1}{\sum m_i} \sum_i m_i \cdot \text{CrossEntropy}(\hat{y}_i, y_i)$$
- Loss mask $m_i = 1$ strictly on assistant completion tokens, $m_i = 0$ on user prompt and padding tokens.
- Verified: User prompt tokens receive zero gradient.

#### Arm B: Relational Token Weighting
$$L_B = \frac{\sum_i w_i \cdot \text{CrossEntropy}(\hat{y}_i, y_i)}{\sum_i w_i}$$
- Weights: $w_i = 2.0$ on relational answer tokens, $w_i = 1.0$ on conversational dialogue tokens, $w_i = 0.0$ on prompt/padding.
- Verified: Weighting formula is numerically normalized by $\sum w_i$, preventing artificial learning rate scaling.

#### Arm C: Counterfactual Pairwise Assignment Objective
$$S(y | x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log P(y_t | x, y_{<t})$$
$$L_{\text{pair}} = \text{softplus}(\text{margin} - (S(y_{\text{pos}} | x) - S(y_{\text{neg}} | x)))$$
$$L_C = L_{\text{LM}} + 0.2 \cdot L_{\text{pair}}$$
- Margin: $0.5$ nats per answer token.
- Verified Properties:
  - When $S_{\text{pos}} - S_{\text{neg}} = +2.0$ nats: $L_{\text{pair}} = 0.201413 \to 0$.
  - When $S_{\text{pos}} - S_{\text{neg}} = -2.0$ nats: $L_{\text{pair}} = 2.578890$ (heavy penalty).
  - Exact token length matching verified across positive and negative candidates.
  - Gradients successfully backpropagate to all parameters ($|\nabla W| = 45.5408$).
