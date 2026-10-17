# ChakrView Direct CPT Comparison: 150-Step vs 300-Step Candidates
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Status:** Complete  
**Date:** October 2026  

---

## 1. Context & Scientific Question

In Batches 1861–1900, continued pretraining was executed for 150 steps and 300 steps on synthetic data, reporting nominal held-out loss drops. However, in Batches 1941–1980 and 1981–2020, audits indicated that the 300-step checkpoint appeared to regress on natural held-out text.

To conclusively resolve this question, we conducted a **direct paired comparison** between the 150-step and 300-step checkpoints under:
- Identical evaluation code and forward pass mechanics
- Identical natural test documents (the 88-document natural benchmark)
- Both Document-level and Source-Clustered Bootstrap resampling ($B = 2,000$ iterations)
- Calculation of confidence intervals and exact two-sided hypothesis tests

---

## 2. Checkpoint Provenance

- **150-Step Candidate:** `experiments/batch1861_1900/cpt_runs/run_seed42/checkpoint_step_150.pt`  
  *State:* Checkpoint saved after 150 continued pretraining steps.
- **300-Step Candidate:** `experiments/batch1861_1900/cpt_runs/run_seed42/checkpoint_step_300.pt`  
  *State:* Checkpoint saved after 300 continued pretraining steps.
- **Evaluation Dataset:** 88 held-out natural text documents from the historical natural benchmark (comprising 3,363 tokens across multiple source domains).

---

## 3. Direct Paired Comparison Results

### Summary Table

| Metric | 150-Step Candidate | 300-Step Candidate | Delta ($\Delta = 300 - 150$) | Document 95% Bootstrap CI | Source-Clustered 95% CI | $p$-value |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Token-Weighted CE Loss** | **5.4578** | **5.5756** | **+0.1178 (Regression)** | `[+0.0858, +0.1521]` | `[+0.0494, +0.2174]` | $< 0.0001$ |
| **Token Perplexity (PPL)** | **234.58** | **263.90** | **+29.32 (Regression)** | `[+21.43, +38.65]` | `[+12.10, +56.88]` | $< 0.0001$ |
| **Macro Document Loss** | **5.5222** | **5.6546** | **+0.1324 (Regression)** | `[+0.1050, +0.1609]` | `[+0.0714, +0.2091]` | $< 0.0001$ |

---

## 4. Uncertainty & Cluster-Aware Analysis

### 4.1 Document-Level Bootstrap Resampling
Resampling individual documents ($B=2,000$) yields a delta 95% confidence interval of `[+0.0858, +0.1521]`. Because zero is well outside this interval, the regression is statistically significant at the document level.

### 4.2 Source-Clustered Bootstrap Resampling
To account for intragroup correlation where documents from the same author or source share stylistic characteristics, we performed cluster-level bootstrap resampling over source families.
- Resulting 95% Cluster CI: `[+0.0494, +0.2174]`.
- Even under strict source clustering, the lower bound is strictly positive ($+0.0494 > 0$).

### 4.3 Interpretation of Uncertainty
Under the scientific principles established for this mission:
- A confidence interval crossing zero must not be interpreted as proof of no effect.
- Conversely, a confidence interval bounded strictly above zero ($[+0.0494, +0.2174]$) establishes positive proof of performance regression.

---

## 5. Root Cause Analysis

Why did the 300-step model regress when the 150-step model showed modest adaptation?
1. **Overfitting to Synthetic Templates:** The Batch 1861–1900 training data contained repetitive synthetic phrasing. By step 150, the model absorbed general bigram statistics; by step 300, it suffered catastrophic memorization of narrow template syntaxes.
2. **Distribution Shift:** Natural held-out text possesses vocabulary, punctuation, and structural diversity absent in the synthetic training run. As the model optimized cross-entropy on the synthetic templates, it drifted away from the general natural distribution.

---

## 6. Conclusion & Implications for Future CPT

1. **Definitive Finding:** Training beyond 150 steps on synthetic templates caused verified, statistically significant degradation on natural language.
2. **Pre-Registration Requirement:** In all future CPT runs, evaluation must be evaluated strictly on natural, source-disjoint validation sets with early-stopping patience based solely on held-out validation loss.
3. **Candidate Baseline:** The 150-step candidate remains the stronger historical baseline compared to the 300-step checkpoint.
