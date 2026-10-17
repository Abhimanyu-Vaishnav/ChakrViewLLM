# CHECKPOINT IDENTITY AUDIT

**Batch:** 2301–2340  
**Focus:** Forensic Investigation of Seed 42 Path and Hash Discrepancies Across Checkpoint Manifests  
**Status:** FULLY RECONCILED  

---

## 1. Executive Summary

A critical discrepancy was investigated: Batch 2261–2300 cited two different paths and SHA-256 hashes for the Seed 42 candidate:
- **Variant A (Historical & Actual Disk Artifact):**
  - Path: `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt`
  - SHA-256: `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`
  - Size: 41,402,267 bytes
  - State: **EXISTS ON DISK, FULLY LOADABLE**
- **Variant B (Phantom Report Entry):**
  - Path: `experiments/batch2141_2180/checkpoints/cpt_seed_42/best_model.pt`
  - SHA-256: `d3268fc43f114c07d3fa8f5c9a75d9e5d4cb05f1542f5674cfffcfe2db894101`
  - State: **DOES NOT EXIST ON DISK**

### Root Cause Analysis
Forensic analysis of the filesystem and Git history proves conclusively that `Variant B` was a typographic hallucination in conversational commentary:
1. Directory `experiments/batch2141_2180/checkpoints/cpt_seed_42/` has never existed. The directory created during Batch 2141–2180 was `experiments/batch2141_2180/cpt/seed_42/`.
2. The hash `d3268fc43f114c07d3fa8f5c9a75d9e5d4cb05f1542f5674cfffcfe2db894101` appears nowhere in the actual serialized files, manifests, or JSON evaluation records. The true SHA-256 computed directly from `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` is `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`.
3. In [candidate_manifest.json](file:///D:/Project/ChakrView/experiments/batch2261_2300/candidate_manifest.json), line 25 already correctly recorded `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`.

---

## 2. Complete Checkpoint Inventory Across All 6 Evaluation Models

Every checkpoint evaluated in the independent benchmark was directly verified on disk:

| Model Identifier | Relative Path | File Size (Bytes) | Computed SHA-256 | Training Step | Checkpoint Format | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 42,212,787 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | 10,000 | State Dict + Opt | **CONFIRMED VALID** |
| **Historical CPT-150** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | 13,794,913 | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | 150 | Model Weights Only | **CONFIRMED VALID** |
| **Historical CPT-300** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | 13,795,727 | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | 300 | Model Weights Only | **CONFIRMED VALID** |
| **CPT Seed 42 Best** | `experiments/batch2141_2180/cpt/seed_42/cpt_best_checkpoint.pt` | 41,402,267 | `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3` | 125 | State Dict + Opt | **CONFIRMED VALID** |
| **CPT Seed 1337 Best**| `experiments/batch2141_2180/cpt/seed_1337/cpt_best_checkpoint.pt` | 41,402,267 | `05c021c37b61f6e6869833a27d46380c3eaa23c062b6aa4341a3c5a755e0f28b` | 125 | State Dict + Opt | **CONFIRMED VALID** |
| **CPT Seed 2026 Best**| `experiments/batch2141_2180/cpt/seed_2026/cpt_best_checkpoint.pt` | 41,402,203 | `c47e826d37279ef6c639cb802dae112adb8d253be9ad38d9b3ebc266af4291f9` | 125 | State Dict + Opt | **CONFIRMED VALID** |

### Proof of Checkpoint Distinctness
All six SHA-256 hashes are unique. No checkpoint is an alias or copy of another. The three CPT candidates (Seed 42, Seed 1337, and Seed 2026) were trained independently and produced distinct parameter distributions and evaluation losses.
