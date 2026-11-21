# CHAKRVIEW RESEARCH ARTIFACT INVENTORY: CV-GENESIS-NR-004

**Mission ID:** CV-GENESIS-NR-004  
**Date:** 21 November 2026  
**Artifact Directory:** `experiments/genesis_canonical_pretraining_004/`  

---

### 1. Tokenizer Artifacts (Canonical Ground Truth)

| Artifact Path | Size | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `data/experiments/vocab_4096/vocab.json` | 112,627 B | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Canonical vocabulary mapping (4,096 tokens) strictly matching Stage-C binary token shards. |
| `data/experiments/vocab_4096/merges.txt` | 44,064 B | `32e54117b07db3d9db7c3760d6ee656f4e1f76da94b15ad47a5fb68393fa1199` | Canonical BPE merges file. |

---

### 2. Pretrained Model Checkpoints

| Artifact Path | Size | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | 20,733,723 B | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Optimal checkpoint saved at Step 1600 (Validation Loss: 2.9744, Validation PPL: 19.58) with verified vocabulary hash. |
| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_final.pt` | 20,733,721 B | `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` | Final checkpoint at Step 1624 (2 complete epochs, 13.3M token exposures). |

---

### 3. Pipeline & Training Scripts

| Artifact Path | Size | Description |
| :--- | :--- | :--- |
| `experiments/genesis_canonical_pretraining_004/train_canonical_pretraining.py` | 7,760 B | Complete GPU pretraining harness with cryptographic tokenizer verification assertions and cosine warmup. |

---

### 4. Comprehensive Mission Reports

| Report Path | Description |
| :--- | :--- |
| `reports/GENESIS_NR_004_GIT_PRESERVATION.md` | Audit and log of Stage 0 Git tree audit and commit `9286592`. |
| `reports/GENESIS_NR_004_TOKENIZER_ALIGNMENT.md` | Verification of the 3,677-token misalignment resolution and canonical SHA-256 match. |
| `reports/GENESIS_NR_004_PRETRAINING_LOG.md` | Complete step-by-step training curves, throughput, and hardware efficiency log. |
| `reports/GENESIS_NR_004_FOUNDATION_EVALUATION.md` | Quantitative validation and generation evaluation comparing old vs new pretraining models. |
| `reports/GENESIS_NR_004_FINAL_REPORT.md` | Executive synthesis across all 5 stages of Mission CV-GENESIS-NR-004. |
