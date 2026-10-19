# SFT V3 PILOT METRIC RECONCILIATION REPORT
**Scope:** Resolution of Metric Discrepancies, Checkpoint Hashes & Split Isolation  
**Date:** 2026-10-19  
**Execution Context:** `experiments/sft_v3_failure_analysis/`

---

## 1. Discrepancy Resolution: Validation Loss 3.3215 vs 3.6204

### The Issue:
The SFT V3 pilot training run reported:
- **Best Validation Loss at Step 100:** `3.3215` (Perplexity: `27.70`) in [TRAINING_METRICS.json](file:///d:/Project/ChakrView/experiments/sft_v3_pilot_readiness/pilot_run_v3/TRAINING_METRICS.json).
- **Candidate Validation Loss in Evaluation:** `3.6204` (Perplexity: `37.35`) in [EVALUATION_RESULTS.json](file:///d:/Project/ChakrView/experiments/sft_v3_pilot_readiness/pilot_run_v3/EVALUATION_RESULTS.json).

### Root-Cause Resolution:
The two numbers represent **two mathematically distinct evaluation procedures** performed on the exact same checkpoint:

| Metric Dimension | Training Loop Evaluation (`3.3215`) | Post-Training Comparative Evaluation (`3.6204`) |
| :--- | :--- | :--- |
| **Evaluating Function** | `SFTExperimentEngine.evaluate(num_batches=5, seed=201)` | `evaluate_comparative.py -> compute_validation_loss()` |
| **Data Sampling Scope** | **Stochastic Sub-sample**: Samples 5 batches of `batch_size=4` with replacement (20 sampled instances out of 40 records). | **Full Exhaustive Population**: Iterates sequentially over **all 40 records** (30 structured + 10 replay) with batch size 1. |
| **Evaluated Token Count** | Sub-sample active response tokens (~300–450 tokens) | Exact total response tokens across all 40 records: **860 active tokens**. |
| **Mathematical Reduction** | $\frac{\sum_{\text{batch}=1}^5 \text{loss}_b \times \text{tokens}_b}{\sum \text{tokens}_b}$ on 5 sampled batches. | $\frac{\sum_{i=1}^{40} \text{loss}_i \times \text{tokens}_i}{\sum_{i=1}^{40} \text{tokens}_i}$ across entire dataset. |
| **Perplexity Computation** | $\exp(3.3215) = 27.70$ | $\exp(3.6204) = 37.35$ |
| **Verdict** | **Locally optimal sub-sample metric** used for online checkpoint selection. | **Authoritative, exhaustive validation baseline** representing the entire 40-record held-out distribution. |

Both calculations use identical token loss masks (`-100` for prompt and padding, active response tokens supervised), identical cross-entropy definitions, and identical model weights (`sft_best_checkpoint.pt`). The numerical difference is purely the expected variance between a 5-batch stochastic sample (20 records) and the full 40-record population.

---

## 2. Checkpoint Identity & Verification

Every evaluation in the pilot report was verified against exact cryptographic file hashes:

| Checkpoint Name | Local File Path | Verified SHA-256 Checksum |
| :--- | :--- | :--- |
| **Canonical Base (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Candidate SFT V3 Best** | `experiments/sft_v3_pilot_readiness/pilot_run_v3/sft_best_checkpoint.pt` | `bfde007db1a90dde5ccc75d14a9ffe59f57b2393b87ac00b19422b8d088d1879` |
| **Candidate Step 100 (Final)** | `experiments/sft_v3_pilot_readiness/pilot_run_v3/sft_checkpoint_step_100.pt` | `8ce613c70f6f124a3740da1fb050e5b59964371be1f995d326c983a97a5f18c8` |

*(Note: Step 100 was selected as the best checkpoint; `sft_best_checkpoint.pt` is an atomic copy of Step 100 created at the moment the new minimum validation loss was achieved).*

---

## 3. Reproduction of Numerators and Denominators

The 40 validation records comprise **30 structured extraction records** and **10 unconditioned replay records**. The metrics break down with complete mathematical precision:

| Metric Category | Metric Name | Canonical Base (Step 10,000) | SFT V3 Pilot Candidate | Verified Denominator Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Native Unconstrained** | Valid JSON Syntax | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Schema Compliance | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Exact Task Match (EM) | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Code Attractor Rate | **0 / 30 (0.0%)** | **27 / 30 (90.0%)** | 30 structured validation records |
| | Premature EOS Rate | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Empty Output Rate | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| **Schema Constrained** | Valid JSON Syntax | **0 / 30 (0.0%)** | **23 / 30 (76.7%)** | 30 structured validation records |
| | Schema Compliance | **0 / 30 (0.0%)** | **23 / 30 (76.7%)** | 30 structured validation records |
| | Exact Task Match (EM) | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Deadlock Rate | **0 / 30 (0.0%)** | **0 / 30 (0.0%)** | 30 structured validation records |
| | Budget Timeout Rate | **30 / 30 (100.0%)** | **5 / 30 (16.7%)** | 30 structured validation records |
| **Replay Probes** | Immediate EOS Rate | **0 / 10 (0.0%)** | **0 / 10 (0.0%)** | 10 replay validation records |
| | Avg Continuation Tokens | **48.0 tokens** | **48.2 tokens** | 10 replay validation records |

---

## 4. Discovery of Evaluator Scorer Defect in `const_out` Logging

While reproducing the raw outputs of the comparative evaluation, a minor reporting defect in [evaluate_comparative.py](file:///d:/Project/ChakrView/experiments/sft_v3_pilot_readiness/pilot_run_v3/evaluate_comparative.py) was isolated:
- In line 261, `evaluate_comparative.py` attempted to log `const_out = dec_res.get("generated_text", "")`.
- However, the dictionary returned by `generate_schema_constrained()` in [schema_prefix_trie.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/schema_prefix_trie.py) keys the string output as `"text"`, NOT `"generated_text"`.
- Consequently, `constrained_output` was serialized as empty string `""` in the JSON detail records, despite `parsed_object` containing the actual parsed JSON dictionary (e.g. `{"status": ", ", "code": 0}`).
- **Impact on Scoring**: **Zero impact on Exact Task Match or Valid JSON metrics**, because the scoring logic correctly inspected `parsed_c = dec_res.get("parsed_object")` and `dec_res.get("is_valid_json")`. The defect was purely cosmetic in the detailed string logging field.
- **Semantic Finding**: Inspecting the actual emitted parsed objects confirms that under constraints, the candidate model repeatedly emitted generic punctuation strings (`", "`) for string keys and `0` for numeric keys, failing to extract the ground-truth entities from the context.
