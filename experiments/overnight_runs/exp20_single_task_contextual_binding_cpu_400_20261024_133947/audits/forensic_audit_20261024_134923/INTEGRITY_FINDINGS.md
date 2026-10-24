# Exp20 Forensic Integrity Findings

**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  
**Audit Timestamp**: `2026-10-24T13:49:23`  

---

## 1. Summary of Cryptographic Hash Checks

Every protected repository asset and Exp20 experimental artifact was independently recomputed and compared against expected SHA-256 signatures:

| Target Asset | Verified Path | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Canonical Starting Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705...` | `96357705...` | **PASS** |
| **Isolated Checkpoint Copy** | `exp20_.../checkpoints/training_checkpoint_start.pt` | `96357705...` | `96357705...` | **PASS** |
| **Architecture Digest (Seed 42)** | `ChakrMicro(ModelConfig())` parameter byte stream | `c5571c9c...` | `c5571c9c...` | **PASS** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119...` | `db4d6119...` | **PASS** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dc...` | `ada4b1dc...` | **PASS** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51...` | `9080ff51...` | **PASS** |
| **Model Source** | `chakrview/brain/model.py` | `1343d7d1...` | `1343d7d1...` | **PASS** |
| **Exp17 Train Split** | `experiments/.../data/train_exp17_v1.jsonl` | `074fafe5...` | `074fafe5...` | **PASS** |
| **Exp17 Val Split** | `experiments/.../data/val_exp17_v1.jsonl` | `15d9c942...` | `15d9c942...` | **PASS** |
| **Exp17 Test Split** | `experiments/.../data/test_exp17_v1.jsonl` | `be10afaa...` | `be10afaa...` | **PASS** |
| **Exp20 Train Dataset** | `exp20_.../data/exp20_train_context_extraction.jsonl` | `9b8a66d9...` | `9b8a66d9...` | **PASS** |
| **Exp20 Val Dataset** | `exp20_.../data/exp20_val_context_extraction.jsonl` | `dc927f9a...` | `dc927f9a...` | **PASS** |
| **Exp20 Test Dataset** | `exp20_.../data/exp20_test_frozen.jsonl` | `497eb8f1...` | `497eb8f1...` | **PASS** |
| **Exp20 Best Checkpoint** | `exp20_.../checkpoints/sft_best_checkpoint.pt` | `8d4ecf9e...` | `8d4ecf9e...` | **PASS** |

---

## 2. Integrity Verdict

- **Canonical Checkpoint Isolation**: The canonical step 10,000 checkpoint was untouched; the experiment trained exclusively from the verified isolated copy.
- **Source Code Preservation**: No modifications were made to `session.py`, `model.py`, or tokenizer files.
- **Overall Integrity**: **100% PASS across all 14 evaluated artifacts**. Recorded in [HASH_VERIFICATION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/HASH_VERIFICATION.json).
