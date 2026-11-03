# CHAKRVIEW — CAND_01 EVALUATOR SPECIFICATION v1.0
## Deterministic Metric Computation, Target Extraction, and Statistical Confidence

**Document ID:** `DOC-EXP-CAND01-EVL-001`  
**Location:** `docs/experiments/CAND_01_EVALUATOR_SPECIFICATION_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 9  
**Status:** `SPECIFICATION_ONLY` (Implementation Plan Only — No Evaluator Execution Authorized)  
**Date:** 2026-10-31  

---

## 1. Evaluator Architecture and Core Principles

The evaluation harness for CAND_01 is engineered to eliminate scoring ambiguity, false positives from unanchored substring searches, and false negatives from delimiter mismatches.

### Mandatory Operational Directives
1. **Zero Unanchored Substring Searching:**
   Constructs like `if target in completion:` are **strictly prohibited**. In small vocabulary distributions, short strings can appear by random chance or in repetitive loops, leading to inflated accuracy.
2. **Exact First-Token / Word-Boundary Anchoring:**
   Because all prompts end with the canonical query delimiter ` -> `, the expected completion begins immediately at token position 0 of the model's generated sequence.
   - For single-token entity values: The evaluator checks whether `generated_tokens[0] == target_token_id`.
   - For textual representations: The evaluator matches using word-boundary regex `^target\b`.
3. **Deterministic Greedy Decoding:**
   All benchmark evaluations are performed under greedy decoding (`temperature=0.0`, `top_k=1`, `top_p=1.0`) to ensure 100% bit-exact reproducibility across runs.

---

## 2. Metric Taxonomy and Mathematical Formulations

### 2.1 Primary Metric: Exact-Match Copy Accuracy ($A_{\text{EM}}$)
The proportion of evaluation instances where the model's generated output exactly matches the true target value:
$$A_{\text{EM}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\hat{y}_i = y_i^*\right)$$
Where $\hat{y}_i$ is the parsed greedy output and $y_i^*$ is the ground-truth entity value.

### 2.2 Secondary Metric: Target Token Probability Rank ($R_{\text{tgt}}$)
Measures the rank of the correct target token in the model's pre-softmax logit distribution at the first response token position (evaluated under teacher-forced prompt conditioning):
$$R_{\text{tgt}} = 1 + \sum_{v \in \mathcal{V}} \mathbb{I}\left(z_v > z_{y^*}\right)$$
- Optimal value: $R = 1$ (target token has highest logit).
- Random baseline: $R \approx 2046$ (for $|\mathcal{V}| = 4093$).
- M5.1 failure benchmark: $R = 205.5$.

### 2.3 Secondary Metric: Prompt Sensitivity Delta ($\Delta p$)
Measures the causal influence of prompt context on output probability by swapping the target value in context ($y_A \to y_B$) while keeping the query key unchanged:
$$\Delta p = \left| P(y_A \mid \text{Prompt}_{A}) - P(y_A \mid \text{Prompt}_{B}) \right|$$
- Target threshold: $\Delta p \ge 0.20$ (indicating strong causal coupling).
- M5.1 failure benchmark: $\Delta p = 0.00014$ (near-zero coupling).

### 2.4 Secondary Metric: Attention Weight Flow ($W_{\text{attn}}$)
Measures the average attention weight from the query position to the context key and value token positions in the final transformer layer (Layer 5):
$$W_{\text{key}} = \frac{1}{H} \sum_{h=1}^H A_{h}[T_{\text{query}}, T_{\text{key}}], \quad W_{\text{val}} = \frac{1}{H} \sum_{h=1}^H A_{h}[T_{\text{query}}, T_{\text{val}}]$$

---

## 3. Granular Error Classification Taxonomy

Every failed test instance ($\hat{y} \neq y^*$) is automatically categorized into one of six mutually exclusive error classes:

| Error Code | Classification Name | Definition / Empirical Pattern | Mechanistic Interpretation |
|---|---|---|---|
| **ERR-01** | **Distractor Intrusion** | $\hat{y} \in \{\text{Distractor Values}\}$ | Model retrieved an irrelevant context value; attention routing failure. |
| **ERR-02** | **Key Echo** | $\hat{y} \in \{\text{Query Key or Distractor Keys}\}$ | Model copied the key instead of the bound value; binding breakdown. |
| **ERR-03** | **Prompt Leakage / Syntax Echo** | $\hat{y} \in \{\text{"PAIR", "QUERY", "PAIRS", "COPY", "->"}\}$ | Attention collapsed onto syntax prompt tokens; template attractor. |
| **ERR-04** | **Repetition Loop** | $\hat{y}$ consists of repeating single characters/tokens | Softmax collapse into high-frequency stationary states. |
| **ERR-05** | **Empty / Immediate EOS** | $\hat{y} = \text{""}$ or first token is `<EOS>` | Premature termination; sequence termination prior to generation. |
| **ERR-06** | **Out-of-Context Generation** | $\hat{y} \notin \{\text{All Context Tokens}\}$ | Unanchored language model prior emission; hallucination. |

---

## 4. Stratified Evaluation Protocol Across Task Families

The evaluator evaluates the test split ($N = 500$) and stratifies metrics across the 8 dimensions:

```
[ EVALUATION SUITE: 500 Test Instances ]
   |-- TF-01: Direct Copy Accuracy (N=62)
   |-- TF-02: Single Pair Accuracy (N=62)
   |-- TF-03: Multi-Pair Accuracy (N=62)
   |-- TF-04: Distractor Curve (N=62 stratified by 1, 3, 7 distractors)
   |-- TF-05: Similar-Key Accuracy (N=62)
   |-- TF-06: Unseen Rebinding Accuracy (N=64)
   |-- TF-07: Distance Decay Curve (N=63 stratified by 10, 50, 100, 200 tokens)
   `-- TF-08: Context Length Curve (N=63 stratified by 64, 128, 256, 400 tokens)
```

---

## 5. Statistical Rigor, Sample Size, and Uncertainty Quantification

### 5.1 Sample Size Justification
- With test sample size $N = 500$, an observed accuracy of $75.0\%$ yields a standard error of:
  $$SE = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.75 \times 0.25}{500}} = \sqrt{\frac{0.1875}{500}} \approx 0.01936 \quad (1.94\%)$$
- The 95% Wilson score confidence interval is approximately $[71.0\%, 78.6\%]$, which easily rules out the null hypothesis boundary ($< 5.0\%$) with $p < 10^{-15}$.

### 5.2 Mandatory 50-Sample Manual Spot-Check Protocol
Before an evaluation run can be signed off:
1. Randomly sample 50 records from the test evaluation output using fixed seed `seed=999`.
2. A human reviewer or independent audit agent must inspect:
   - Prompt string vs. input tokens.
   - Raw generated output vs. parsed output.
   - Ground truth target string vs. recorded metric score.
3. If there is a single discrepancy ($> 0$ scoring errors in 50 samples), the entire evaluation run is classified as **`EVALUATOR_DEFECT`** and rejected.

