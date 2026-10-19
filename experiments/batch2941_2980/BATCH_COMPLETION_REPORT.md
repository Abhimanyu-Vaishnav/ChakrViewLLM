# ChakrView Batch 2941–2980 Completion Report
## Balanced SFT Implementation, Best-Checkpoint Saving & Regression-Safe Preflight

### Executive Summary

Batch 2941–2980 successfully implemented and verified the infrastructure required to execute a balanced SFT experiment designed to alleviate the raw-completion `<EOS>` regression discovered in Batch 2901–2940. 

All primary objectives and acceptance gates have been satisfied:
1. **Best-Checkpoint Saving:** Implemented in `chakrview/training/sft_engine.py` with atomic write-and-replace, deterministic tie-breaking, and metadata tracking.
2. **Balanced Dataset Builder:** Built 400-example dataset with a 70% instruction (280 examples across 7 categories) and 30% replay (120 natural pretraining continuations) mixture, split 80/10/10 into train (320), val (40), and test (40).
3. **Replay & Instruction Masking:** Audited and implemented in `chakrview/training/prompt_contract.py` with `append_eos=False` support for natural continuations, removing artificial `<EOS>` supervision on pretraining text.
4. **Preflight Audit:** Created and verified `run_preflight.py` passing all 6 integrity gates.
5. **Zero Training Invariant:** Verified that zero optimizer steps, fine-tuning, or model weight mutations occurred.
6. **Safety Invariants:** Canonical Step-10,000 checkpoint and tokenizer artifacts confirmed bit-identical; `chakrview/runtime/session.py` remains completely untouched.

---

### Acceptance Gate Audit

| Gate | Requirement | Status | Evidence / Verification |
| :--- | :--- | :--- | :--- |
| **Gate A** | Best-checkpoint saving implemented & tested | **PASS** | `chakrview/training/sft_engine.py` implements atomic write via `.tmp` file and replaces `sft_best_checkpoint.pt`. Tested in `tests/test_balanced_sft_batch2941_2980.py::test_01`. |
| **Gate B** | Deterministic dataset construction & provenance | **PASS** | `experiments/batch2941_2980/build_balanced_dataset.py` uses seed 42, records SHA-256 in `DATASET_MANIFEST.json`. |
| **Gate C** | Replay and instruction label masks explicitly tested | **PASS** | Tested in `tests/test_balanced_sft_batch2941_2980.py::test_02`. Replay omits terminal `<EOS>`, instruction preserves terminal `<EOS>`. |
| **Gate D** | Baseline and SFT step-100 compared under shared protocol | **PASS** | Recorded in `experiments/batch2941_2980/BASELINE_REGRESSION_RESULTS.json`. |
| **Gate E** | Raw-completion regression measured rather than hidden | **PASS** | Documented in `EOS_AND_LABEL_AUDIT.md`: 15/28 (53.6%) immediate `<EOS>` rate in step-100 candidate vs 0/28 in baseline. |
| **Gate F** | Preflight validates checkpoint & tokenizer identities | **PASS** | `experiments/batch2941_2980/run_preflight.py` verifies SHA-256 hashes and configurations, writes `PREFLIGHT_REPORT.json` (`PASS`). |
| **Gate G** | Zero training or optimizer steps occurred | **PASS** | Verified across all executions; preflight refuses training without explicit flag. |
| **Gate H** | Canonical checkpoint and tokenizer hashes unchanged | **PASS** | Step-10,000 SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` verified bit-identical. |
| **Gate I** | `chakrview/runtime/session.py` untouched | **PASS** | Confirmed via `git status` — file was never staged, opened for write, or modified. |
| **Gate J** | Existing user changes preserved | **PASS** | Modified files `verified_tool_router.py`, `integrated_assistant_adapter.py`, `assistant_cli.py`, `test_conversation_memory.py` intact. |

**Overall Batch Verdict: PASS**

---

### Files Modified & Created

#### Source Code Changes (Minimal & Deliberate)
- `chakrview/training/prompt_contract.py`: Added `append_eos: bool = True` parameter to `build_causal_sequence` to support natural continuation replay without forced `<EOS>` targets.
- `chakrview/training/sft_engine.py`:
  - Added `save_best_checkpoint: bool = True` to `SFTConfig`.
  - Added `best_val_loss` and `best_step` tracking in `SFTExperimentEngine`.
  - Added atomic safe-replace pattern for `sft_best_checkpoint.pt`.
  - Added deterministic tie-breaking on equal validation losses (strictly earlier checkpoint retained).
  - Added `append_eos` support in `SFTDataset.encode_example` and `get_batch`.

#### New Batch Artifacts (`experiments/batch2941_2980/`)
- `build_balanced_dataset.py`: Generator script creating the 400-example 70/30 mixture.
- `BALANCED_SFT_DATASET.jsonl`: Complete 400-record dataset (280 instruction / 120 replay).
- `train.jsonl`: 320 training examples (80%).
- `val.jsonl`: 40 validation examples (10%).
- `test.jsonl`: 40 held-out test examples (10%).
- `DATASET_DESIGN.md`: Detailed dataset schema, mixture design, and category breakdown.
- `DATASET_MANIFEST.json`: File hashes, split counts, and provenance metadata.
- `EOS_AND_LABEL_AUDIT.md`: In-depth causal loss and terminal-token audit explaining the raw-completion regression.
- `BASELINE_REGRESSION_RESULTS.json`: Consolidated evaluation comparison between baseline and Step-100 candidate.
- `SFT_EXPERIMENT_CONFIG.json`: Fully specified training configuration for the proposed balanced SFT pilot.
- `run_preflight.py`: Preflight verification engine with 6 critical integrity gates.
- `PREFLIGHT_REPORT.json`: Preflight verification results recording clean PASS.
- `NEXT_TRAINING_PROPOSAL.md`: Hyperparameter, schedule, stopping criteria, and runtime estimates for the next authorized run.
- `BATCH_COMPLETION_REPORT.md`: This completion report.

#### Test Suites Added
- `tests/test_balanced_sft_batch2941_2980.py`: Unit and integration tests for best-checkpoint saving, tie-breaking, replay masking, dataset balance, and preflight verification.

---

### Test Verification

1. **SFT Infrastructure Test Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/test_balanced_sft_batch2941_2980.py tests/test_sft_experiment_preparation.py
   ```
   **Result:** `13 passed in 3.41s`

2. **Runtime Regression Test Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/test_neural_runtime_adapter.py tests/test_calculator_routing_hardening.py tests/test_conversation_memory.py
   ```
   **Result:** `49 passed in 4.12s`

**Total Passing Tests:** 62 passed, 0 failed.

---

### Baseline vs. Step-100 Regression Findings

As documented in `BASELINE_REGRESSION_RESULTS.json`:
- **Raw-Completion Immediate `<EOS>` Rate:** 
  - Baseline (Step 10,000): **0/28 (0.0%)**
  - SFT Pilot Candidate (Step 100): **15/28 (53.6%)**
- **JSON Format Adherence:**
  - Baseline: **0/3 (0.0%)**
  - SFT Step 100: **2/3 (66.7%)**
- **QA Exact Match:**
  - Baseline: **0/24 (0.0%)**
  - SFT Step 100: **0/24 (0.0%)**

Fine-tuning on 100% instruction data taught the model to produce `<EOS>` whenever a sentence ends (`.`, `?`), destroying raw completion. The new 70/30 balanced dataset with `append_eos=False` on replay examples directly provides gradient pressure against spurious terminal `<EOS>` generation.

---

### Invariant and Security Checks

1. **`chakrview/runtime/session.py` status:**
   `git diff --name-only chakrview/runtime/session.py` returned empty (completely untouched).
2. **Canonical Checkpoint Hash:**
   `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Unchanged).
3. **Canonical Tokenizer Artifacts:**
   `release/chakrview-v0.5.0/data/experiments/vocab_4096/tokenizer.json` (Unchanged).
4. **Pre-existing Working-Tree Modifications:**
   `chakrview/runtime/verified_tool_router.py` (Preserved).
   `chakrview/runtime/integrated_assistant_adapter.py` (Preserved).
   `chakrview/runtime/assistant_cli.py` (Preserved).
   `tests/test_conversation_memory.py` (Preserved).
5. **Training Execution Status:** Zero optimizer steps or background jobs were launched.

---

### Known Limitations & Next Action

- **Model Capacity:** ChakrMicro v0.1 has 2,510,400 parameters and context window 512. While 70/30 balance is hypothesized to preserve continuation while maintaining JSON structure, true multi-task generalization remains constrained by parameter scale.
- **Authorization Required:** To execute the 160-step balanced training pilot, the operator must explicitly review [NEXT_TRAINING_PROPOSAL.md](file:///d:/Project/ChakrView/experiments/batch2941_2980/NEXT_TRAINING_PROPOSAL.md) and execute:
  ```powershell
  .\.venv\Scripts\python.exe experiments/batch2941_2980/run_preflight.py --execute-authorized-training
  ```
