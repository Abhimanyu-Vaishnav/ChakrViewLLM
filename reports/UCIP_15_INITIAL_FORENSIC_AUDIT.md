# UCIP-15 Initial Forensic Audit: Repository, Environment & Checkpoint Integrity

**Mission ID:** UCIP-15  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Execution Agent:** Antigravity  
**Audit Timestamp:** 2026-11-20T14:13:00+05:30  
**Audit Status:** PASSED (All Forensic Invariants Verified)  

---

## 1. Git Repository & Commit Lineage Audit

The active branch and commit history were inspected prior to any modifications:
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current HEAD Commit:** `25dfd4bcae64cbf073f2490a52b0e7b2b6ce9d18` (`docs(release): prepare ChakrView conversation model release package`)
- **Parent Commit:** `f11f2d8cfe3c63911b936567c14d4576e0af73b2` (`feat(ucip-14r): build and independently audit conversation dataset`)
- **Grandparent Commit:** `6af23bb54fec9c1ffe21453d2fdf577c912555c0` (`feat(ucip14cr2): complete final causal target alignment audit...`)
- **Working Tree Cleanliness:** Verified no uncommitted changes to tracked files. Uncommitted untracked legacy experiment scratch files in `experiments/cognition/ucip_05/` remain isolated and untouched.

---

## 2. Invariant & Checkpoint Cryptographic Hashes

All non-negotiable invariants were verified via direct SHA-256 computation:

| Artifact Target | Expected SHA-256 | Measured SHA-256 | Status |
|---|---|---|:---:|
| **Runtime Session** (`chakrview/runtime/session.py`) | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH ($\Delta W = 0$)** |
| **Canonical Baseline** (`instantiate_frozen_baseline`) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| **Base Checkpoint** (`checkpoint_step_800.pt`) | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |
| **Tokenizer Vocab** (`vocab.json`) | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH** |
| **Tokenizer Merges** (`merges.json`) | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH** |

---

## 3. Dataset Artifact Hashes & Partition Counts

| Partition File | Expected Records | Measured Records | SHA-256 Checksum | Status |
|---|---:|---:|---|:---:|
| `data/sft/ucip14r/sft_train_12000.jsonl` | 12,000 | 12,000 | `2c634a9e2cb5e1d7101dd41ac51b5884c308d120b33ac3ad759a94701bfe4cd6` | **MATCH** |
| `data/sft/ucip14r/sft_validation_1500.jsonl` | 1,500 | 1,500 | `a16fcd872e925683cc402b4da26c4abb13a77bbc9e9b24c8a46a1304a015acf7` | **MATCH** |
| `data/sft/ucip14r/sft_dev_1500.jsonl` | 1,500 | 1,500 | `86108cc72fd923335010d408dacce51333e98adea263a8f08839fc6f0a24ff12` | **MATCH** |
| `data/sft/ucip14r/sft_master_15000.jsonl` | 15,000 | 15,000 | `726ded9108211f0eec6d2c9ca5fac452e811c521c46014fc9fb8f4461168db6b` | **MATCH** |
| `experiments/ucip14r/benchmarks/sealed/benchmark_ucip14_sealed_500.json` | 500 | 500 | `dcbd7f221c1c54d27c9b00a3dbd4f43fee0b79b9fb124f6a86fa6069650da8ea` | **MATCH** |

---

## 4. Hardware Environment & Runtime Specifications

- **Operating System:** Windows 11 Pro (Build 10.0.26300)
- **Python Version:** Python 3.14.7 (64-bit AMD64)
- **PyTorch Version:** PyTorch 2.14.0+cpu (CPU execution target)
- **System Memory:** 31.64 GB total, 13.56 GB available
- **Storage:** 203.29 GB total on drive D:\, 81.78 GB available free space
- **Hardware Acceleration:** CUDA is unavailable; execution is 100% CPU-native local execution.

---

## 5. Pretrained Base Checkpoint State Verification

- **Checkpoint Step:** 800
- **Reported Pre-Training Validation Loss:** 4.6859
- **Reported Pre-Training Test Loss:** 4.6595
- **Pretrained Baseline Stage C Validation Loss:** Measured at **3.9544** on 20 held-out Stage C chunks (10,240 tokens). This serves as the retention anchor for Gate E.

---

## 6. Audit Conclusion
Phase 0 forensic audit confirms 100% integrity across all code, checkpoints, datasets, and hashes. No artifacts have been tampered with or corrupted. Proceeding to Phase 1.
