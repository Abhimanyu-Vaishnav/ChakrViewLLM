# Phase F & G — R2 Baseline Evaluation & Statistical Reporting

**Date:** 2026-11-18  
**Mission:** UCIP-10E-R2 Clean Benchmark Reconstruction, Evaluator Repair & Baseline Re-evaluation  
**Suite:** `benchmark_r2_sealed.json` (460 Genuinely Held-Out Probes, 0% Contamination)  
**Evaluator:** `StrictEvaluatorsR2`  
**Decoding:** Temperature 0.0, Max tokens 48, Rep penalty 1.05, No-repeat 3-gram: Yes  
**Raw Results:**
- UCIP-09: `experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json`
- UCIP-09R: `experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json`
- UCIP-09S Pilot: `experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json`

---

## 1. Clean Benchmark Capability Matrix (R2 Sealed Suite)

| Capability Domain | Total Probes | UCIP-09 Medium (Primary Baseline) | UCIP-09R Medium (Degraded) | UCIP-09S Condition C (Balanced Pilot) | 95% Wilson Confidence Interval (UCIP-09) |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Language Intelligence** | 75 | **0 / 75 (0.0%)** | 0 / 75 (0.0%) | 0 / 75 (0.0%) | $[0.00\%, 4.88\%]$ |
| **2. Mathematics** | 75 | **0 / 75 (0.0%)** | 0 / 75 (0.0%) | 0 / 75 (0.0%) | $[0.00\%, 4.88\%]$ |
| **3. Exact Copying** | 50 | 0 / 50 (0.0%) | 0 / 50 (0.0%) | 0 / 50 (0.0%) | $[0.00\%, 7.13\%]$ |
| **4. Logical Reasoning** | 50 | **10 / 50 (20.0%)** | **10 / 50 (20.0%)** | **10 / 50 (20.0%)** | $[11.24\%, 32.74\%]$ |
| **5. Generalization (OOD)** | 60 | 0 / 60 (0.0%) | 0 / 60 (0.0%) | 0 / 60 (0.0%) | $[0.00\%, 6.00\%]$ |
| **6. Multi-Turn Reasoning** | 40 | 0 / 40 (0.0%) | 0 / 40 (0.0%) | 0 / 40 (0.0%) | $[0.00\%, 8.81\%]$ |
| **7. Self-Correction** | 30 | 0 / 30 (0.0%) | 0 / 30 (0.0%) | 0 / 30 (0.0%) | $[0.00\%, 11.35\%]$ |
| **8. Uncertainty / Abstention** | 30 | 0 / 30 (0.0%) | 0 / 30 (0.0%) | 0 / 30 (0.0%) | $[0.00\%, 11.35\%]$ |
| **9. Coding Intelligence** | 30 | 0 / 30 (0.0%) | 0 / 30 (0.0%) | 0 / 30 (0.0%) | $[0.00\%, 11.35\%]$ |
| **10. Runtime Awareness** | 20 | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) | $[0.00\%, 16.11\%]$ |
| **TOTAL ACCURACY** | **460** | **10 / 460 (2.17%)** | **10 / 460 (2.17%)** | **10 / 460 (2.17%)** | **$[1.19\%, 3.96\%]$** |

---

## 2. In-Depth Scientific Analysis of the R2 Results

### Why Did Language Drop from 45% in Old 10E to 0.0% in R2?
This is the single most revealing scientific finding of the entire UCIP-10 series:
1. **The Old 10E Language Probes Were Contaminated Memorization Probes:**
   In the old 10E benchmark, 45% of language probes passed because prompts like `"Hello ChakrView. How are you?"`, `"What does a CPU do?"`, and `"Explain the difference between RAM and storage."` were **literal copies of the training set** (`ucip09_train.jsonl`). The model had memorized the exact responses during SFT.
2. **On Genuinely Unseen Language Prompts, The Model Fails:**
   When presented with novel systems questions (e.g. `"What is the function of the instruction register inside a CPU?"`, `"Describe how pipelining improves instruction throughput."`), the model generates ungrounded tokens and corrupted bi-grams:
   - Example output: `'The express software instructions from memory and exchange where the the the arithmetic and except t t of the many.'`
3. **The Reality of 12.9M Small Causal Decoders:**
   A 12.9M parameter model trained on 3,240 SFT examples **does not possess general language intelligence or technical comprehension**. It possesses a lookup memory for the specific sentence stems present in its SFT dataset. When tested on genuinely held-out, non-contaminated prompts, its true language generalization score is **0.0%**.

### Why Did Logic Score 20.0% Across All Models?
The only probe category that passed in R2 was `SEA_LOG_002` (and its copies):
`"All squares are rectangles. Shape S is a square. Is Shape S a rectangle? Answer with 'Yes' or 'No'."` $\to$ Generated `'Yes.'`
Because the model has a default bias toward affirmative responses (`'Yes.'`) on binary queries, it correctly answered the affirmative syllogism probes while failing all negative or open-ended logic problems.
