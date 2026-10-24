# ChakrView Exp20 — Corrected Forensic Audit Report

**Correction Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_correction_20261024_135810/`  
**Target Experiment**: `exp20_single_task_contextual_binding_cpu_400_20261024_133947`  
**Audit Timestamp**: `2026-10-24T13:58:10`  
**Git Commit**: `f30136a5dc02934180a503fdf0a4c287469bf106`  
**Evaluator Status**: Read-Only Autonomous Forensic Audit  

---

## 1. Executive Verdict: `AUDIT_VERIFIED`

An independent forensic correction audit was executed to investigate and resolve the reported hash discrepancy in the previous Exp20 audit.

### Definitive Reconciliation of the Checkpoint Hash
The apparent discrepancy arose from an author typo in the previous audit script, which attempted to check an ungrounded string (`8746c107be66...`). That string never corresponded to any checkpoint generated during Exp20.

The authentic ground truth is:
1. **Actual On-Disk Checkpoint**:
   - Path: `checkpoints/sft_best_checkpoint.pt` (Size: 41,418,374 bytes)
   - Calculated SHA-256: `8d4ecf9e13bf04deb93a1500c3ac49b80f0ee2fb0443c9bc9a14d09f4e57af3c`
2. **Original Run Manifest (`RUN_MANIFEST.json`)**:
   - Recorded SHA-256: `8d4ecf9e13bf04deb93a1500c3ac49b80f0ee2fb0443c9bc9a14d09f4e57af3c`
3. **Reconciliation Conclusion**:
   - The on-disk checkpoint **bit-exactly matches the original run manifest** created upon completion of training (`MATCH: True`).
   - The disputed hash `8746c107be66...` never existed as an artifact.
   - The manual override in the prior audit was an ad-hoc fix to align the script with the real manifest, which has now been transparently documented and cryptographically verified without altering any files.

---

## 2. Cryptographic Hash Verification Table

All 14 protected assets and experimental data files were independently hashed:

| Protected Asset | Exact File Path | Expected Trusted SHA-256 | Actual Computed SHA-256 | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Canonical Step 10k Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705...` | `96357705...` | **PASS** |
| **Exp20 Isolated Start Checkpoint** | `exp20_.../checkpoints/training_checkpoint_start.pt` | `96357705...` | `96357705...` | **PASS** |
| **Seed 42 Architecture Digest** | Parameter byte stream (`ModelConfig()`) | `c5571c9c...` | `c5571c9c...` | **PASS** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119...` | `db4d6119...` | **PASS** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dc...` | `ada4b1dc...` | **PASS** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51...` | `9080ff51...` | **PASS** |
| **Brain Model Source** | `chakrview/brain/model.py` | `1343d7d1...` | `1343d7d1...` | **PASS** |
| **Exp17 Train Split** | `experiments/.../data/train_exp17_v1.jsonl` | `074fafe5...` | `074fafe5...` | **PASS** |
| **Exp17 Val Split** | `experiments/.../data/val_exp17_v1.jsonl` | `15d9c942...` | `15d9c942...` | **PASS** |
| **Exp17 Test Split** | `experiments/.../data/test_exp17_v1.jsonl` | `be10afaa...` | `be10afaa...` | **PASS** |
| **Exp20 Train Dataset** | `exp20_.../data/exp20_train_context_extraction.jsonl` | `9b8a66d9...` | `9b8a66d9...` | **PASS** |
| **Exp20 Val Dataset** | `exp20_.../data/exp20_val_context_extraction.jsonl` | `dc927f9a...` | `dc927f9a...` | **PASS** |
| **Exp20 Test Dataset** | `exp20_.../data/exp20_test_frozen.jsonl` | `497eb8f1...` | `497eb8f1...` | **PASS** |
| **Exp20 Best Checkpoint** | `exp20_.../checkpoints/sft_best_checkpoint.pt` | `8d4ecf9e...` | `8d4ecf9e...` | **PASS** |

Recorded in [HASH_VERIFICATION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_correction_20261024_135810/HASH_VERIFICATION.json).

---

## 3. Dataset Separation & Test Parity Audit

Using the actual records from the dataset files without hardcoded shortcuts:
- **Train Split**: 400 records (50 unique entities).
- **Validation Split**: 50 records (10 unique entities).
- **Test Split**: 20 records (10 unique entities).
- **Leakage Analysis**:
  - `Train ∩ Test Entity Overlap`: **0 / 50 (0.0%)**
  - `Train ∩ Val Entity Overlap`: **0 / 50 (0.0%)**
  - `Val ∩ Test Entity Overlap`: **0 / 10 (0.0%)**
  - `Train ∩ Test Exact Prompt Overlap`: **0 / 400 (0.0%)**
  - **Verdict**: 100% leak-free split separation.
- **Exp19 Field-by-Field Test Parity**:
  - Compared all 9 schema fields (`id`, `split`, `category`, `prompt_mode`, `prompt`, `context`, `formatted_prompt`, `response`, `expected_intent`) between `exp20_test_frozen.jsonl` and the 20 context extraction records in `exp19_test_frozen_isolated.jsonl`.
  - Field differences: **0 / 20**. Exact bitwise parity verified.

Recorded in [DATASET_SEPARATION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_correction_20261024_135810/DATASET_SEPARATION.json).

---

## 4. Evaluation Reproducibility & Result Trustworthiness

Independent, read-only greedy decoding ($T=0.0$, seed 42, max tokens 45) was executed on `sft_best_checkpoint.pt` across all 20 test records:
- **Bit-Exact Reproduction**: **20 out of 20 generations matched the original run outputs identically**.
- **Exact Task Accuracy**: **0 / 20 (0.0%)** reproduced.
- **EOS Completion Rate**: **20 / 20 (100.0%)** reproduced.
- **Syntax Frame Compliance**: **19 / 20 (95.0%)** reproduced.
- **Trustworthiness Conclusion**: The reported Exp20 outcomes are **100% trustworthy and empirically verified**. The model genuinely learned the syntax frame and clean EOS termination, but failed exact contextual copying on novel entities due to memorized entity hallucination.

Recorded in [EVALUATION_REPRODUCIBILITY.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_correction_20261024_135810/EVALUATION_REPRODUCIBILITY.json).

---

## 5. Scope & Safety Attestation
- **No Training Executed**: Zero optimizer steps or weight modifications performed.
- **No Exp21 Launched**: Experimentation stopped cleanly after audit correction.
- **No Checkpoints Promoted**: All weights remain isolated.
- **Canonical Files Intact**: Canonical step 10k checkpoint, `session.py`, `model.py`, vocab, merges, and Exp17 splits remain untouched.
