# CHAKRVIEW RESEARCH ARTIFACT INVENTORY: CV-GENESIS-NR-005

**Mission ID:** CV-GENESIS-NR-005  
**Date:** 23 November 2026  
**Artifact Directory:** `experiments/genesis_conversational_foundation_005/`  

---

### 1. Tokenizer Artifacts (Canonical Ground Truth)

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Description | Git Status |
| :--- | :--- | :--- | :--- | :--- |
| `data/experiments/vocab_4096/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Canonical vocabulary mapping (4,096 tokens). | Tracked (`23aac68`) |
| `data/experiments/vocab_4096/merges.json` | 76,299 | `b68010ba086111f185efb1fa85fb5e197d1656fa47eec03dcae324a4f89d53f3` | Canonical merges table. | Tracked (`23aac68`) |

---

### 2. Conversational SFT Datasets

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Record Count | Git Status |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/genesis_conversational_foundation_005/data/train.jsonl` | 1,180,682 | `7048ca...` | 2,901 | Staged |
| `experiments/genesis_conversational_foundation_005/data/validation.jsonl` | 148,460 | `0f0bf6...` | 362 | Staged |
| `experiments/genesis_conversational_foundation_005/data/test.jsonl` | 149,850 | `476a6d...` | 364 | Staged |
| `experiments/genesis_conversational_foundation_005/data/dataset_manifest.json` | 2,140 | `e5bc41...` | Summary | Staged |

---

### 3. Model Checkpoints (Git LFS)

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Description | Git Status |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | 20,733,723 | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Foundation Pretrained Checkpoint | Tracked LFS (`da2e00e`) |
| `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 62,203,157 | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Conversational SFT Best Checkpoint | Staged LFS |
| `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt` | 62,203,387 | `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0` | Conversational SFT Final Checkpoint | Staged LFS |

---

### 4. Implementation Scripts & Deliverables

| Artifact Path | Size (Bytes) | Description |
| :--- | :--- | :--- |
| `experiments/genesis_conversational_foundation_005/build_sft_dataset.py` | 13,875 | Workstream A dataset extraction, audit, and balancing pipeline. |
| `experiments/genesis_conversational_foundation_005/train_sft.py` | 10,250 | Workstream B masked SFT training pipeline with hash gates. |
| `experiments/genesis_conversational_foundation_005/evaluations/run_100_prompt_benchmark.py` | 12,480 | Workstream C 100-prompt comparative benchmark harness. |
| `experiments/genesis_conversational_foundation_005/evaluations/evaluate_multiturn.py` | 5,420 | Workstream C multi-turn context retention evaluation. |
| `experiments/genesis_conversational_foundation_005/application_candidate/app.py` | 7,850 | Workstream D isolated candidate HTTP server on port 7865. |
