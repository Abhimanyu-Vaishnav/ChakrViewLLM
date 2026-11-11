# CR-EXP-16F: Evaluator Audit & Verification Report

## 1. Scope & Benchmark Integrity

The evaluation suite was executed via `evaluate_cr16f.py` across all twelve trained checkpoints. Each model was evaluated under greedy autoregressive decoding ($T=0$) against six benchmark suites:

1. **D1. CR-16 Copying Benchmark (100 probes):**
   - Source: `experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json` (exact sequence copying category).
   - SHA-256: `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97`.
   - Criteria: Exact string match of entire output sequence against target.
2. **D2. CR-16C Sealed Key-Value Retrieval (1,500 probes):**
   - Source: `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json`.
   - Partition: 1,400 Answerable Probes (CAT1–CAT8, CAT10) and 100 Unanswerable Probes (CAT9).
   - Metrics: Answerable exact match, False Abstention rate (`pred == "[UNKNOWN]"` on answerable), and Unanswerable Recall (`pred == "[UNKNOWN]"` on CAT9).
3. **D3. Benchmark B1 Position Invariance (600 probes):**
   - Source: `cr16e_b1_position_invariance_600.json`.
   - Evaluates key-value extraction across positions 0, 1, 2, and 3.
4. **D3b. Benchmark B2 Permutation & Counterfactual Pairs (150 + 150 pairs):**
   - Source: `cr16e_b2_permutation_pairs_150.json` & `cr16e_b2_counterfactual_pairs_150.json`.
   - Metric: Joint consistency (both probes within the pair must be strictly correct).
5. **D4. Benchmark B3 Multi-Token Spans (500 probes):**
   - Source: `cr16e_b3_multitoken_span_500.json`.
   - Evaluates multi-token entity completion across lengths 1, 2, 3, and 4 tokens.
6. **D5. Benchmark B4 Calibrated Abstention (300 probes):**
   - Source: `cr16e_b4_abstention_300.json`.
   - Metric: Abstention recall (`pred == "[UNKNOWN]"`).
7. **D6. CR-16C CAT7 Unseen-Value Generalization (150 probes):**
   - Evaluates zero-shot extraction of entity values never seen during retrieval training.

---

## 2. Evaluation Output Auditing

Every prediction for each of the 12 models was recorded in `eval_outputs/<run_name>_eval.json` with prompt, target, model prediction, and boolean match flags.
Master summary metrics are compiled in `cr16f_evaluation_master_summary.json`.

All metrics reported below were computed directly from the saved JSON prediction logs with no cherry-picking, post-filtering, or manual overrides.

