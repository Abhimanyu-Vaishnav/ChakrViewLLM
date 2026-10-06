# Step 165 — Objective & Optimization Investigation

## 1. Experimental Methodology

Step 165 compared three controlled objective formulations on isolated ChakrMicro candidates:
- **EXP_A:** Standard causal next-token cross-entropy across all prompt and target tokens.
- **EXP_B:** Answer-position weighted causal loss ($5\times$ weight assigned to the target token).
- **EXP_C:** Reasoning-target focused loss (gradients evaluated strictly on the answer prediction position).

---

## 2. Comparative Matrix

| Objective | Description | Train Loss Initial | Train Loss Final | Held-Out Reasoning Acc | Language Retention Loss | Mean Grad Norm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_A** | Standard Causal Cross-Entropy | 8.3510 | 1.9421 | 0.0000 | 7.9120 | 0.4210 |
| **EXP_B** | Weighted Causal Loss (5x) | 8.4120 | 1.5420 | 0.5000 | 8.0120 | 0.6512 |
| **EXP_C** | Target-Focused Loss | 8.3980 | **0.8421** | **0.5000** | **8.0410** | **0.5120** |

---

## 3. Analysis & Tradeoffs

1. **Syntax Sinks Gradients in EXP_A:** In standard causal next-token prediction, $>80\%$ of token positions in reasoning prompts are syntactic boilerplate (`order`, `:`, `>`, `->`, `first`). Minimizing loss on these tokens exhausts gradient steps without enforcing the relational copying logic.
2. **Efficiency of EXP_C:** Target-focused loss forces all parameter updates to directly improve target logit ranking, reducing final target loss to $0.8421$ and enabling non-zero template generalization while preserving language retention ($8.0410 < 8.25$).
