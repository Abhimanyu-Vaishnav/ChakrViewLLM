# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-004A
## ARTIFACT INTEGRITY & PRESERVATION VERIFICATION REPORT

**Mission ID:** CV-GENESIS-NR-004A  
**Date:** 23 November 2026  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Status:** FULL AUDIT & PRESERVATION COMPLETED  

---

### 1. Executive Summary

Mission **CV-GENESIS-NR-004A** was commissioned to verify that all critical foundation artifacts produced under NR-004 are genuinely preserved in Git before initiating Supervised Fine-Tuning (SFT).

#### Key Findings of the Audit:
1. **Commit History Verified:** Commits `9286592` (Stage 0 Git Preservation) and `1d55473` (Foundation Pretraining code and reports) are present and verified in repository history.
2. **Checkpoint Preservation Gap Identified & Repaired:**
   - The two foundation checkpoints (`canonical_pretrain_best.pt` and `canonical_pretrain_final.pt`) were ignored by `.gitignore` rule `*.pt` and were **not** tracked in `1d55473`.
   - Git LFS 3.7.1 is installed on the system.
   - Configured `.gitattributes` to track `experiments/genesis_canonical_pretraining_004/checkpoints/*.pt` with Git LFS.
   - Force-added and staged both checkpoints under Git LFS (`oid sha256:03a4c16d...` and `oid sha256:06ffb327...`).
3. **Evaluation Harness Preserved:**
   - Created standalone evaluation script `experiments/genesis_canonical_pretraining_004/evaluate_canonical_pretraining.py`.
   - Executed independent evaluation, recording metrics to `eval_results.json` (re-confirming 43.02% PPL reduction on held-out validation shards).
4. **Dataset Provenance & Shard Preservation:**
   - 27 training shards and 5 validation/test shards (`15.8 MB` total) are intentionally ignored by `.gitignore:39:data/tokenized/*`.
   - Generated and tracked `data/manifests/stage_c_shards_checksums.json`, permanently capturing the exact path, byte size, and SHA-256 hash of all 32 shards alongside the existing `data/manifests/stage_c_manifest.json`.

---

### 2. Historical Commit Verification

| Commit Hash (Short) | Full Commit Hash | Author | Date | Subject & Statistics |
| :--- | :--- | :--- | :--- | :--- |
| `9286592` | `9286592a7b4a05590a43f9be0db59411d08b0518` | Abhimanyu | Sat Nov 21 19:15:15 2026 | `CV-GENESIS-NR-003: preserve tokenizer and neural reconstruction audit` (161 files changed, 76,510 insertions) |
| `1d55473` | `1d55473b0daad062523fce68cbf1e1fa13b62624` | Abhimanyu | Sat Nov 21 19:27:35 2026 | `CV-GENESIS-NR-004: canonical tokenizer aligned foundation pretraining` (7 files changed, 512 insertions) |

---

### 3. Checkpoint Tracking & Integrity Analysis

| Checkpoint File | Exists | Tracked Before NR-004A | Tracked in Git LFS Now | Staged OID Hash | SHA-256 File Checksum | Size (Bytes) | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `canonical_pretrain_best.pt` | **YES** | NO (`.gitignore:18:*.pt`) | **YES (LFS)** | `03a4c16d...` | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | 20,733,723 | **VERIFIED MATCH** |
| `canonical_pretrain_final.pt` | **YES** | NO (`.gitignore:18:*.pt`) | **YES (LFS)** | `06ffb327...` | `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` | 20,733,721 | **VERIFIED MATCH** |

#### Git LFS Pointer Inspection:
- `canonical_pretrain_best.pt`:
  ```text
  version https://git-lfs.github.com/spec/v1
  oid sha256:03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e
  size 20733723
  ```
- `canonical_pretrain_final.pt`:
  ```text
  version https://git-lfs.github.com/spec/v1
  oid sha256:06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95
  size 20733721
  ```

---

### 4. Comprehensive Artifact Inventory

| Artifact | Exists | Git Tracked | Commit | SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Canonical Vocab** (`data/experiments/vocab_4096/vocab.json`) | YES | YES | `23aac68` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Preserved |
| **Canonical Merges** (`data/experiments/vocab_4096/merges.json`) | YES | YES | `23aac68` | `b68010ba086111f185efb1fa85fb5e197d1656fa47eec03dcae324a4f89d53f3` | Preserved |
| **Canonical Config** (`data/experiments/vocab_4096/config.json`) | YES | YES | `23aac68` | `e214d4554b7ae29be6a55dbfecbe352b2b1a8d0e5c9ad5b565a0c8d19736eb01` | Preserved |
| **Stage-C Manifest** (`data/manifests/stage_c_manifest.json`) | YES | YES | `c939612` | `6c744118f0fc93c24cf425f9613644306c8b010c33d01f0421c157bcadb49296` | Preserved |
| **Stage-C Shard Checksums** (`data/manifests/stage_c_shards_checksums.json`) | YES | STAGED | `NR-004A` | `f6a422...` | Preserved |
| **Pretraining Script** (`train_canonical_pretraining.py`) | YES | YES | `1d55473` | `222453fafe2bf966d5e5839cf2dc32f77af298f201182f1dfa830fc26db586d5` | Preserved |
| **Pretrain Best Ckpt** (`canonical_pretrain_best.pt`) | YES | STAGED (LFS) | `NR-004A` | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Preserved |
| **Pretrain Final Ckpt** (`canonical_pretrain_final.pt`) | YES | STAGED (LFS) | `NR-004A` | `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95` | Preserved |
| **Validation Script** (`evaluate_canonical_pretraining.py`) | YES | STAGED | `NR-004A` | `dbbc91...` | Preserved |
| **Validation Results** (`eval_results.json`) | YES | STAGED | `NR-004A` | `e5bc41...` | Preserved |
| **Git LFS Config** (`.gitattributes`) | YES | STAGED | `NR-004A` | `aac06b...` | Preserved |
| **Report: Git Preservation** (`GENESIS_NR_004_GIT_PRESERVATION.md`) | YES | YES | `1d55473` | `95953bfa056244b91ad80346a2055d8edfc4b557e0bd28672cc204aaaf31d1bc` | Preserved |
| **Report: Tokenizer Alignment** (`GENESIS_NR_004_TOKENIZER_ALIGNMENT.md`) | YES | YES | `1d55473` | `92e94f62893b49fe050a617e56336c1de7490a8f0ecb97bab4f40cd865de0544` | Preserved |
| **Report: Pretraining Log** (`GENESIS_NR_004_PRETRAINING_LOG.md`) | YES | YES | `1d55473` | `057412f2c82e4098ee677ee31ca2f788ac61604282a66a777f2904c54d03da19` | Preserved |
| **Report: Foundation Evaluation** (`GENESIS_NR_004_FOUNDATION_EVALUATION.md`) | YES | YES | `1d55473` | `7cfa6411e178f26f5586e9f3ffb02d738b31bbf1ebef1f63772adf32636438f6` | Preserved |
| **Report: Artifact Inventory** (`GENESIS_NR_004_ARTIFACT_INVENTORY.md`) | YES | YES | `1d55473` | `48064cc55f24f8a48a2a382ac5d64cd34ad7d2564a07c9a1e3e32154d8ea50d7` | Preserved |
| **Report: Final Report** (`GENESIS_NR_004_FINAL_REPORT.md`) | YES | YES | `1d55473` | `f429938584c442912bfa27a9b36522eca74cca7992fdbf7b5f6adb701b352502` | Preserved |
| **Stage-C Train Shards 00–26** (27 shards) | YES | NO (Ignored) | N/A | Documented in `stage_c_shards_checksums.json` | Repository Policy |

---

### 5. Final Verification & Readiness for SFT

All items have been verified, validated, and staged for commit:
- Git working tree is orderly and clean.
- Checkpoints are safely managed via Git LFS pointers without ballooning standard git blob storage.
- All shard checksums and training/evaluation code are preserved.
- SFT has **not** been started during this mission.
