# CR-EXP-01: REVISED EVALUATION & ABLATION PLAN

This plan defines the comprehensive evaluation battery, statistical reporting standards, confidence interval formulations, and mechanistic ablation methods for CR-EXP-01.

---

## 1. Quantitative Reporting Standards

All evaluations report:
1. **Exact-Match Retrieval Accuracy ($EM$)**:
   $$\text{EM} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{Predicted} == \text{Target})$$
2. **Invalid Output Rate ($IOR$)**:
   Percentage of outputs that fail to produce a valid token, produce an unformatted delimiter, or loop repeatedly.
3. **95% Confidence Intervals (Wilson Score Interval)**:
   For sample size $N$ and observed accuracy $\hat{p}$:
   $$\text{CI}_{95} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}} \quad (z = 1.96)$$

---

## 2. Condition-by-Condition Evaluation Suite

| Test File | Condition Code | Primary Metric | Baseline Threshold | Target Acceptance Gate |
|---|---|---|---|---|
| `test_familiar_100.jsonl` | **Condition A (Familiar)** | Exact Match ($EM_A$) | $\ge 0.0\%$ | $\ge 90.0\%$ |
| `test_novel_100.jsonl` | **Condition B (Novel In-Context)** | Exact Match ($EM_B$) | $\ge 0.0\%$ | $\ge 60.0\%$ |
| `test_distractor_sweep_100.jsonl` | **Condition D (Distractor Sweep)** | Accuracy vs $k \in \{1..4\}$ | $\ge 0.0\%$ | $\ge 50.0\%$ at $k=3$ |
| `test_multiturn_100.jsonl` | **Condition G (Multi-Turn Recall)** | Exact Match ($EM_G$) | $\ge 0.0\%$ | $\ge 70.0\%$ |
| `test_unseen_template_100.jsonl` | **Condition I (Unseen Syntax)** | Exact Match ($EM_I$) | $\ge 0.0\%$ | $\ge 55.0\%$ |
| `step102_syntactic_probes.json` | **Regression Gate (Syntax)** | Top-5 Accuracy | 15.0% (Frozen Baseline) | $\ge 15.0\%$ (No regression) |
| `UNIVERSAL_CAPABILITY_BENCHMARK`| **Regression Gate (General)** | Target Hits | 21.1% (Parent Baseline) | $\ge 21.1\%$ (No regression) |

---

## 3. Mechanistic Ablation Analysis

Ablation analysis is explicitly separated from behavioral accuracy scores:

1. **Attention Map Tracing**:
   - For novel test sequences ($N=20$), extract attention maps $A^{(l, h)} \in \mathbb{R}^{T \times T}$ across layers $1\dots 6$ and heads $1\dots 6$.
   - Calculate average attention weight from the query key token position to the context value token position.
2. **Causal Head Ablation Probe**:
   - Zero out candidate head $(l, h)$ activations during forward pass.
   - Re-evaluate $EM_B$ on novel associations.
   - If zeroing $(l, h)$ causes a statistically significant degradation ($\Delta EM_B \ge 30\%$) while leaving general English perplexity unaffected, the head is formally designated an **Induction Head**.
   - If no individual head shows this causal impact, the report will state that binding is distributed across the multi-head subspace rather than localized.
