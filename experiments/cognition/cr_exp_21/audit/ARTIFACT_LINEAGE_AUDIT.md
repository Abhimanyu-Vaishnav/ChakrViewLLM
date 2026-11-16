# CR-EXP-21: Phase A — Repository Recovery & Artifact Lineage Audit

**Mission ID:** CR-21  
**Phase:** Phase A — Repository Recovery & Lineage Audit  
**Date:** 2026-11-15  
**Investigator:** Antigravity Autonomous Coding Agent  
**Repository Working Copy:** `D:\Project\ChakrView`  
**Git Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Cryptographic Audit of Protected & Canonical Artifacts

All mandatory immutable artifacts were audited with bit-exact SHA-256 checks:

| Artifact Name | Path / Definition | Expected SHA-256 | Actual Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Architecture Init** | `torch.manual_seed(42)` param buffer | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (Bit-Exact)** |
| **Canonical Stage B Checkpoint** | `experiments/batch16_21_combined/...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (Bit-Exact)** |
| **Frozen CR-16 Benchmark** | `experiments/.../cr16_synthetic_benchmark_1000.json` | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97` | **PASS (Bit-Exact)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Bit-Exact)** |
| **Conversational Parent** | `experiments/cbr01_sft/.../sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Bit-Exact)** |
| **v0.9.1 Release Checkpoint** | `releases/v0.9.1_conversational_preview/...` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **PASS (Bit-Exact)** |
| **v0.10 Release Checkpoint** | `releases/v0.10_conversational_model/...` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | **PASS (Bit-Exact)** |
| **v0.10.1 Starting Candidate** | `releases/v0.10.1_conversational_model/chakrview_v0101_model.pt` | `548b20a73d3cb3a662bcbfe74ab4f108d7bfdc4c60e56abd76049b260fa16a3b` | `548b20a73d3cb3a662bcbfe74ab4f108d7bfdc4c60e56abd76049b260fa16a3b` | **PASS (Bit-Exact)** |

Zero protected artifacts have been modified or violated.

---

## 2. Checkpoint Verification: `chakrview_v0101_model.pt`

- **File Size:** $41,390,531$ bytes
- **State Dictionary Check:** 57 tensor buffers.
- **Tied Embeddings & Parameter Count:**
  - Unique trainable parameter count: **3,443,136 parameters**.
  - Total state_dict tensors: $4,229,568$ (difference of exactly $786,432 = 4,096 \times 192$, reflecting tied token embeddings and lm_head projection).
- **Lineage Metadata:**
  - Parent Checkpoint: `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` (v0.10 seed 2026).
  - Training Step: 350.
  - Final Validation Loss: $2.7878$ (Validation Token Accuracy: $48.90\%$).
  - Training Seed: 4040.
  - Optimizer State: Complete AdamW state dictionary preserved.

---

## 3. Dataset Lineage & Historical Count Discrepancy Forensic

Section 3 required investigating the discrepancy between CR-20 reported counts (1,419 train, 157 validation) and the filenames `conversation_train_1080.jsonl` / `conversation_val_120.jsonl`.
- **Finding:** Inspection confirms that the files located in `experiments/conversation/cr_exp_17/data/` were named during an early planning phase (`_1080` and `_120`), but were subsequently augmented during CR-17 to contain **exactly 1,419 training lines** and **157 validation lines**.
- The CR-20 training dataset (`cr_exp_20/data/train.jsonl`) contains 940 dialogues (705 synthetic multilingual + 235 replay).
- The CR-20R recovery dataset (`cr_exp_20r/data/recovery_train.jsonl`) contains 1,000 dialogues (748 multilingual recovery + 252 replay).

Lineage is completely reconstructed and mathematically verified. Phase A is complete.

