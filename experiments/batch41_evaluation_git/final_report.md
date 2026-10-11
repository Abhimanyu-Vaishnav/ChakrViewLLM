# CHAKRVIEW — BATCH 41 MASTER REPORT
## Evaluation Reliability, Runtime/Training Alignment, and Verified Git Commits

**Sprint Date:** 2026-10-11  
**Working Branch:** `dev/v0.6-v0.7-validation`  
**Initial HEAD:** `7c3be923acf997d98f693b3509bece86ddf97641`  
**Final HEAD:** `b271cf0` (5 verified commits)  
**Tracked Working Tree:** Clean, except preserved user change `chakrview/runtime/session.py`.

---

## 1. Executive Summary & Final Gate Statuses

Batch 41 achieved three critical milestones:
1. Resolved the fundamental evaluation and formatting discrepancy between the SFT training loop (plain text) and the runtime session (ChatML `<|im_start|>` / `<|im_end|>`).
2. Established deterministic, full-dataset evaluation without random batch subsampling.
3. Cleaned and structured accumulated work across Batches 33–41 into 5 verified, atomic Git commits while keeping `chakrview/runtime/session.py` isolated and uncommitted.

| Gate | Status | Evidence / Notes |
| :--- | :--- | :--- |
| **`EVALUATION_RELIABLE`** | **PASS** | Implemented non-subsampled, deterministic evaluation across all held-out validation tokens with exact loss and perplexity aggregation. |
| **`TRAINING_RUNTIME_ALIGNED`** | **PASS** | Aligned SFT training format with runtime `ConversationTemplate.format_conversation`. Verified ChatML tokenization decomposes to valid frozen subwords. |
| **`SAFETY_VALIDATED`** | **PASS** | Engine-level batch safety validated: all-masked batches and out-of-bounds token IDs are rejected fail-closed before forward/loss computation. |
| **`BASELINE_PRESERVED`** | **PASS** | Canonical initialization hash (`c5571c9...`) and Step 10,000 checkpoint hash (`9635770...`) confirmed 100% bit-exact. |
| **`COMMITS_VERIFIED`** | **PASS** | 5 focused commits created on `dev/v0.6-v0.7-validation`. Full test suite of 49 batch tests passes in 3.68s. |
| **`MAIN_MERGE_READY`** | **NOT_READY** | Branch must not be merged into `main`: `session.py` remains uncommitted pending user sign-off, and qualitative generation remains repetitive over short SFT steps. |

---

## 2. Phase A & B: Evaluation Reliability & Defect Resolution

### Defects Identified in Prior Evaluation
1. **Subsampling Variance**: `SFTEngine.evaluate()` sampled only 5 random batches (up to 10 examples) with a fixed seed, which artificially skipped certain examples in small validation sets.
2. **Format Asymmetry**: Batch 40 trained on raw strings (`user: ...\nassistant: ...`), whereas the runtime session formatted conversations using ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). When the runtime invoked the model, the model encountered 31 unfamiliar tokens at the sequence boundary, corrupting generation.

### Corrections Implemented
- **Deterministic Full Evaluation**: Evaluates the complete held-out validation dataset (all 7 examples, 195 active tokens) without sampling or skipping.
- **Aligned Causal Targets**: Properly shifts next-token targets such that every prompt token is masked (`-100`) and the assistant response (including the closing `<|im_end|>\n` and EOS `<EOS>`) is supervised.

---

## 3. Phase C & D: Runtime/Training Alignment & Experiment Reassessment

### Tokenizer Properties of ChatML Delimiters
- Tokenizer: Frozen BPE, vocabulary size 4096.
- `<|im_start|>` tokenizes to 8 subwords: `[63, 127, 1012, 98, 1832, 119, 127, 65]`.
- `<|im_end|>` tokenizes to 8 subwords: `[63, 127, 1012, 98, 275, 103, 127, 65]`.
- All subwords fall strictly within $[0, 4095]$. No out-of-vocabulary token IDs are required.

### Empirical Reassessment (Base Model vs. Batch 40 vs. Batch 41 Aligned)

| Model Condition | Dataset Format | Full Val Loss | Val Perplexity | Tokens Supervised |
| :--- | :--- | :--- | :--- | :--- |
| **Base Step 10k** | Raw Format (Batch 40) | 7.4067 | 1647.05 | 139 |
| **Batch 40 Step 20** | Raw Format (Batch 40) | 6.7699 | 871.20 | 139 |
| **Base Step 10k** | Aligned ChatML (Batch 41) | 7.4781 | 1768.81 | 195 |
| **Batch 41 Aligned Step 20** | Aligned ChatML (Batch 41) | **6.0563** | **426.79** | 195 |

### Generation Behavior
Under deterministic greedy generation on 5 fixed probes:
- In the aligned setting, cross-entropy loss dropped substantially from **7.48 to 6.06** (a decrease of 1.42).
- However, greedy generation across all 5 probes still degenerates into pretraining wiki table/arithmetic boilerplate (`"|- \n| scope=..."`).
- **Honest Conclusion**: 20 gradient steps on 39 examples is sufficient to improve target token likelihoods, but insufficient to establish coherent instruction-following generation. Capability improvement remains **INCONCLUSIVE / NOT DEMONSTRATED**.

---

## 4. Phase E & F: Verified Git Commits

Five logical, test-backed commits were created on `dev/v0.6-v0.7-validation`:

1. **`ffcda17`**: `feat(training): integrate fail-closed batch safety validation in SFTEngine`
   - File: `chakrview/training/sft_engine.py`
   - Test: `tests/test_sft_engine.py`, `tests/test_batch40_capability_sprint.py`
2. **`46d1538`**: `feat(sft): add EOS collapse diagnostics, token recovery, and regression tests (Batches 33-36)`
   - Files: `experiments/batch33-36/`, `tests/test_batch33-36.py`
   - Test: 13 passed in 1.1s
3. **`1282268`**: `feat(dataset): add grounded corpus extraction pipeline, safety guard, and audit hardening (Batches 37-39)`
   - Files: `experiments/batch37-39/`, `tests/test_batch37-39.py`
   - Test: 29 passed in 2.2s
4. **`13e9ea0`**: `feat(experiment): add Batch 40 continuation training manifests, reports, and tests`
   - Files: `experiments/batch40_capability_sprint/`, `tests/test_batch40_capability_sprint.py`
   - Test: 3 passed in 2.6s
5. **`b271cf0`**: `feat(evaluation): add ChatML format alignment, deterministic full-dataset evaluation, and Batch 41 tests`
   - Files: `experiments/batch41_evaluation_git/`, `tests/test_batch41_evaluation_git.py`
   - Test: 4 passed in 1.8s

### Preservation of User Modification
- `chakrview/runtime/session.py` remains modified and **uncommitted**. It was excluded from all staging operations.

---

## 5. Phase G: Merge-Readiness Assessment

**Status: NOT READY FOR MAIN MERGE**

### Reasons:
1. **Uncommitted User Changes**: `chakrview/runtime/session.py` requires separate user review and validation before it can be merged to `main`.
2. **Dataset Scale**: The grounded dataset has 46 clean examples (39 train, 7 val). Additional domain extractors (dialogues, procedures, mathematics) are required to scale to $\ge 100$ examples.
3. **Generation Quality**: Greedy generation remains bound to pretraining repetitive attractors over 20 steps. Longer, scheduled training on an expanded dataset is required before claiming production conversational capability.
