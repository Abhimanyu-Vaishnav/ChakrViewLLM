# AUDIT ADDENDUM: EXPERIMENT 31 PROTOCOL, SAMPLE COUNTS & DATASET LEAKAGE

**Date:** 2026-10-27
**Auditor:** Principal ML Research Engineer & Experimental Auditor
**Audit Target:** `experiments/exp31_capability_generalization/`
**Classification:** `VALID_WITH_DOCUMENTED_PROTOCOL_DEVIATION`

---

## 1. Executive Summary
Experiment 31 was audited independently prior to the design and execution of Experiment 32.
The audit inspected the preregistration file, protocol, dataset generation script, execution runner, serialized test splits, raw output predictions in `RESULTS.json`, and regression outputs.

The audit verified:
1. **Sample Count Reconciliation:** The actual test set across Groups A through F contains **350 examples** (50, 50, 50, 50, 100, 50), which matches `generate_dataset.py`, `dataset_manifest.json`, `run_exp31.py`, and `RESULTS.json` with bit-exact consistency.
2. **Preregistration Timing:** The pre-registered hash `c2ddd4c8408b352d6ab951d14561063f5df927f45a91243005a6ebb3cef077db` was generated and recorded **before** training/eval execution began, as confirmed by file timestamps and transcript execution logs. The preregistration document itself explicitly states the 350-example breakdown (Group A: 50, Group B: 50, Group C: 50, Group D: 50, Group E: 50 pairs / 100 evals, Group F: 50).
3. **Statistical Claims:** The reported significance of Arm 2 (Pointer) vs. Arm 0/1 on Group B ($10/50$ vs. $0/50$) was independently recomputed via exact one-sided Fisher's Exact Test. The exact $p$-value is $p = 5.934 \times 10^{-4}$ ($p < 0.001$), confirming the mathematical validity of the reported significance claim.
4. **Discovered Protocol Deviations & Leakage:**
   - **Train/Test Group A Prompt Collision:** Exactly 1 prompt out of 50 in `test_group_A_familiar.jsonl` collided with a prompt in `train.jsonl` (`Context: Stream delta: Value: J. Stream gamma: Value: L.\nQuestion: What is the delta value?\nAnswer: `). Because the generation script sampled keys and values uniformly at random with replacement across splits without cross-split set subtraction, 1 in-distribution prompt leaked.
   - **Internal Duplication:** 1 prompt was duplicated within `train.jsonl` (119 unique prompts across 120 records), and 1 prompt was duplicated within `test_group_F_negative_abstain.jsonl` (49 unique prompts across 50 records).
   - **Group B Held-Out Integrity:** Strict zero-leakage held for Group B (Unseen Token Identities), with zero prompt or token overlap with the training set.

---

## 2. Sample-Count & File Manifest Audit

| Split File | Record Count | Unique Prompts | SHA-256 Checksum | Manifest Status |
|---|:---:|:---:|---|:---:|
| `train.jsonl` | 120 | 119 (1 dup) | `7eb36a26c9f4b66b7896756cced7cc2ea49a31d1c27e8ce46a7183a9381ed2e9` | Verified |
| `val.jsonl` | 40 | 40 | `786580513cb8ff8ba4c72e77b6df32b4272046a0f418f360513373200ba46b2b` | Verified |
| `test_group_A_familiar.jsonl` | 50 | 50 | `f17e84a8d3cdb7eac51230891a618e6bced799e39fc9d9f3915a90e505d1a21f` | Verified (1 train overlap) |
| `test_group_B_unseen_tokens.jsonl` | 50 | 50 | `9baeca94bf8f0e6fe1ebb3ae79e77689d80a060761176821cce98a695b3f6d39` | Verified (0 leakage) |
| `test_group_C_permutations.jsonl` | 50 | 50 | `8e3d48219c14f7c8139b3a9e11883e68b855f8b81fa4c1fd9b12d502712bf503` | Verified (0 leakage) |
| `test_group_D_distractors.jsonl` | 50 | 50 | `299665f4c0c3289056d869452b4301feb676f563dfdbab5b52cb679bc45b9ff3` | Verified (0 leakage) |
| `test_group_E_counterfactuals.jsonl` | 100 | 100 (50 pairs) | `c49e12b10d164e370e58cb67db000238d90be0f082f0280a64553ea41481da96` | Verified (0 leakage) |
| `test_group_F_negative_abstain.jsonl` | 50 | 49 (1 dup) | `6219255fe6eea05c5484c405e9fc389f77c0368592aef24b543feab61d32943b` | Verified (0 leakage) |
| **Total Test Examples** | **350** | **349 unique** | — | **Verified** |

---

## 3. Evaluation Contract & Tokenization Audit
- **Prompt Isolation:** `run_exp31.py` evaluates via `x = torch.tensor([[0] + p_toks])`. The target token and `[EOS]` are strictly omitted during greedy autoregressive evaluation.
- **Decoding Parity:** All models were evaluated under identical greedy single-token argmax decoding at prompt termination ($x_{last}$).
- **Target Single-Token Verifiability:** Every key, familiar value, unseen value, and abstention symbol `'?'` maps to a single byte BPE token ID in the canonical vocabulary.
- **Root Failure on Group F:** In Exp31, neither Arm 1 (Canonical SFT) nor Arm 2 (ChakrPointerMicro) was exposed to missing-key examples or abstention targets during training. Consequently, testing on Group F was effectively a pure out-of-domain evaluation, guaranteeing 0.0% accuracy.

---

## 4. Final Audit Classification
- **Classification:** **`VALID_WITH_DOCUMENTED_PROTOCOL_DEVIATION`**
- **Justification:** The primary experimental claim—that `ChakrPointerMicro` provides a statistically significant copying mechanism for out-of-vocabulary tokens ($p = 5.93 \times 10^{-4}$) where autoregressive vocabulary heads completely fail—is robust and reproducible. However, the train-to-test Group A prompt collision and lack of abstention training examples must be resolved in Experiment 32.
