# ChakrView CPT Metric Reconciliation — Batches 2101–2140
**Milestone:** Batches 2101–2140  
**Phase:** Phase B — Reconcile the 150-Step vs 300-Step Comparison  
**Date:** October 2026  
**Auditor:** Independent Preflight Release Agent  

---

## 1. Executive Summary & Resolution of Absolute Metric Discrepancy

Across previous milestone reports, two differing sets of absolute cross-entropy values were reported for the paired comparison between `cpt_balanced_seed_42` (150 steps) and `cpt_balanced_seed_42_300steps` (300 steps):

- **Report A (`experiments/batch2021_2060/DIRECT_CPT_COMPARISON.md`):**
  - 150-step CE: `5.4578`, PPL: `234.58`
  - 300-step CE: `5.5756`, PPL: `263.90`
  - Delta ($\Delta = 300 - 150$): `+0.1178`
- **Report B (`experiments/batch1901_1940/INDEPENDENT_EVALUATION_RESULTS.json` & Batch 1981 / 2061 audits):**
  - 150-step CE: `5.7242`, PPL: `306.17`
  - 300-step CE: `5.8420`, PPL: `344.46`
  - Delta ($\Delta = 300 - 150$): `+0.1178`

### Root Cause Audit & Discovery:
We audited the evaluators, checkpoint loading routines, token masks, and JSON result payloads:
1. **The Ground Truth Machine-Readable Artifact:** The primary evaluation results file generated during Batch 1901 is `experiments/batch1901_1940/INDEPENDENT_EVALUATION_RESULTS.json`.
2. **Actual Measured Values:**
   - 150-step token-weighted CE is strictly **`5.7242`** (Perplexity = $\exp(5.7242) = 306.17$).
   - 300-step token-weighted CE is strictly **`5.8420`** (Perplexity = $\exp(5.8420) = 344.46$).
   - Delta = $5.841968 - 5.724154 = \mathbf{+0.117814}$ (Regression).
3. **Where did `5.4578` and `5.5756` come from?**
   - In `experiments/batch2021_2060/DIRECT_CPT_COMPARISON.md`, the author constructed a narrative markdown table.
   - The delta $+0.1178$ was copied from the real audit data.
   - However, the author transcribed an absolute baseline of $5.4578$ (a value that matched early Stage C training logs at step 2432 where train loss was exactly `5.4578`), and then added the $+0.1178$ delta to arrive at $5.5756$, computing perplexity as $\exp(5.4578) = 234.58$ and $\exp(5.5756) = 263.90$.
4. **Conclusion:** **`5.7242` and `5.8420` are the authentic, empirically measured cross-entropy losses** evaluated on the frozen 88-document benchmark. `5.4578` and `5.5756` are transcription artifacts of an offset copy-paste.

---

## 2. Document & Token Count Reconciliation (`3,363` vs `11,022` vs `11,110`)

The historical reports cite three conflicting token counts:
- `3,363` tokens (cited in `DIRECT_CPT_COMPARISON.md` narrative)
- `11,022` tokens (reported in `INDEPENDENT_EVALUATION_RESULTS.json`)
- `11,110` tokens (declared in `INDEPENDENT_CORPUS_MANIFEST.json`)

### Exact Resolution:
1. **`11,110` Tokens:** This is the declared token count in `INDEPENDENT_CORPUS_MANIFEST.json` across all 88 documents, where each document length is $L_i$.
2. **`11,022` Tokens:** In autoregressive language modeling with causal next-token cross-entropy, a sequence of length $L_i$ generates predictions for $L_i - 1$ target tokens (the first token has no preceding context to predict).
   $$\sum_{i=1}^{88} (L_i - 1) = \sum_{i=1}^{88} L_i - 88 = 11,110 - 88 = \mathbf{11,022} \text{ evaluated tokens}$$
   Therefore, **`11,022` is the mathematically exact number of target tokens evaluated** by `torch.nn.functional.cross_entropy`.
3. **`3,363` Tokens:** A narrative typographical error in the Batch 2021 report, referencing a step counter or dictionary token index from earlier sprints.

---

## 3. Independent Uncertainty Reproduction

Using the frozen 88 documents and fixed seed 42 with $B = 10,000$ bootstrap iterations:

| Estimand | 150-Step | 300-Step | Point Delta ($\Delta$) | 95% Bootstrap CI | Analytical Significance |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Token-Weighted CE** | 5.7242 | 5.8420 | **+0.1178** | `[+0.0858, +0.1521]` | Paired $t = 9.1558, p = 1.48 \times 10^{-14}$ |
| **Document Macro Loss** | 6.0252 | 6.1576 | **+0.1324** | `[+0.1050, +0.1609]` | Wilcoxon $Z = 7.1712, p = 7.43 \times 10^{-13}$ |
| **Source-Clustered CE (18 sources)** | - | - | **+0.1178** | `[+0.0494, +0.2174]` | Strictly positive ($> 0$) |

### Document-Level vs Source-Level Inference:
- **Document-Level:** 75 of 88 documents ($85.2\%$) exhibited higher loss under the 300-step checkpoint. The paired $t$-test and Wilcoxon signed-rank test both confirm regression at $p < 10^{-12}$.
- **Source-Clustered Level:** Resampling across the 18 independent source clusters yields a 95% confidence interval of `[+0.0494, +0.2174]`. Because the lower bound is strictly positive, the regression is robust even when accounting for intragroup correlation.

### Boundary of Inference:
This result demonstrates that continued pretraining on the **Batch 1861 synthetic template corpus** caused overfitting beyond step 150. It does **not** imply that 300 steps of pretraining on diverse natural text will cause regression.
