# CHAKRVIEW RESEARCH ARTIFACT INVENTORY: CV-GENESIS-NR-007

**Mission ID:** CV-GENESIS-NR-007  
**Date:** 24 November 2026  
**Repository:** `D:\Project\ChakrView`  

---

### 1. Canonical Foundation Checkpoints & Tokenizer (Git LFS)

| Artifact Path | Size (Bytes) | SHA-256 Checksum | Description | Git Status |
| :--- | :--- | :--- | :--- | :--- |
| `data/experiments/vocab_4096/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Canonical BPE vocabulary (4,096 tokens). | Tracked (`23aac68`) |
| `data/experiments/vocab_4096/merges.json` | 76,299 | `b68010ba086111f185efb1fa85fb5e197d1656fa47eec03dcae324a4f89d53f3` | Canonical BPE merges table. | Tracked (`23aac68`) |
| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | 20,733,723 | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Foundation Pretrained Checkpoint | Tracked LFS (`da2e00e`) |
| `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_final.pt` | 20,733,721 | `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` | Foundation Pretrained Final Checkpoint | Tracked LFS (`da2e00e`) |
| `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 62,203,157 | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Conversational SFT Best Checkpoint | Tracked LFS (`fef3410`) |
| `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt` | 62,203,387 | `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0` | Conversational SFT Final Checkpoint | Tracked LFS (`fef3410`) |

---

### 2. Validated SFT Datasets

| Dataset Path | Size (Bytes) | SHA-256 Checksum | Count | Status |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/genesis_conversational_foundation_005/data/train.jsonl` | 1,180,682 | `7048ca33ef146ec6947ecda48fca8c65f9038d1796d424b9173fec892e666bc9` | 2,901 | Tracked (`374bbbd`) |
| `experiments/genesis_conversational_foundation_005/data/validation.jsonl` | 148,460 | `0f0bf6085a85ae7518d8441113264663e2aee50b8686f0592964e5261da46542` | 362 | Tracked (`374bbbd`) |
| `experiments/genesis_conversational_foundation_005/data/test.jsonl` | 149,850 | `476a6d0ca0d1cb902f5a6b09bb457e4e04051a88b56dcae32d6ba94d93d50896` | 364 | Tracked (`374bbbd`) |
| `experiments/genesis_conversational_foundation_005/data/dataset_manifest.json` | 2,140 | `e5bc41...` | Summary | Tracked (`374bbbd`) |

---

### 3. Independent Blind Evaluation Benchmark

| Benchmark Path | Size (Bytes) | SHA-256 Checksum | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| `reports/NR006_BLIND_DATASET_MANIFEST.json` | 54,820 | `e2a144b...` | 300 unseen evaluation prompts across 8 subcategories. | Tracked (`ab46a4a`) |
| `reports/NR006_PURE_NEURAL_RESULTS.json` | 231,450 | `d601b3c...` | Complete comparative predictions and scores for all 3 models. | Tracked (`ab46a4a`) |
| `experiments/genesis_intelligence_validation_006/robustness/robustness_results.json` | 8,920 | `73f2a8...` | Results of perturbation, typo, and long-prompt tests. | Tracked (`ab46a4a`) |
| `experiments/genesis_intelligence_validation_006/application/app_verification_results.json` | 5,640 | `1a4e21...` | 10-prompt live HTTP server verification results. | Tracked (`ab46a4a`) |

---

### 4. Non-Committed / Ignored Shards Tracking

| Directory | Shard Count | Total Size | Checksum Manifest Location |
| :--- | :--- | :--- | :--- |
| `data/tokenized/stage_c/train/` | 27 shards | 13.3 MB | `data/manifests/stage_c_shards_checksums.json` (Tracked in Git) |
| `data/tokenized/stage_c/validation/` | 2 shards | 1.0 MB | `data/manifests/stage_c_shards_checksums.json` (Tracked in Git) |
| `data/tokenized/stage_c/test/` | 3 shards | 1.5 MB | `data/manifests/stage_c_shards_checksums.json` (Tracked in Git) |
