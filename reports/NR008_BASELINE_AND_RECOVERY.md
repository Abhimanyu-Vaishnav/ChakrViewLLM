# CV-GENESIS-NR-008 Baseline and Recovery Audit

## 1. Executive Summary

Mission **CV-GENESIS-NR-008** initiates the six-month master development campaign to advance ChakrView from an experimental conversational prototype (`v0.12.0-prototype`) to a robust, independently trained conversational platform (targeting `v1.0` by May 2027).

This report certifies the baseline integrity of the repository, confirms exact cryptographic matching of all foundation artifacts, validates the local hardware envelope, and defines the initial state prior to architectural exploration.

---

## 2. Repository & Remote Baseline

- **Repository Root:** `D:\Project\ChakrView`
- **Current Git Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current Git Commit:** `941e7fcfbf36b042b61f821aec094c9315acdb27`
- **Commit Message:** `CV-GENESIS-NR-007: master technical documentation, release audit, and architecture roadmap`
- **Remote URL:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Remote Synchronization:** In sync with `origin/exp/cbr-01-conversational-breakthrough`.
- **Working Tree State:** Clean; 0 uncommitted changes.

---

## 3. Cryptographic Verification of Base Artifacts

Each critical artifact required for foundation continuity has been verified via SHA-256:

1. **Canonical BPE Vocabulary (4,096 tokens):**
   - Path: `data/experiments/vocab_4096/vocab.json`
   - SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
   - Status: **VERIFIED MATCH**

2. **Canonical Pretraining Best Checkpoint:**
   - Path: `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt`
   - SHA-256: `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e`
   - Status: **VERIFIED MATCH**

3. **Canonical Pretraining Final Checkpoint:**
   - Path: `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_final.pt`
   - SHA-256: `06ffb32722c9a39879ec8b83ae1f4a03ab5ca952253444d06eb2b8297c69aa95`
   - Status: **VERIFIED MATCH**

4. **NR-005 SFT Best Checkpoint:**
   - Path: `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt`
   - SHA-256: `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269`
   - Status: **VERIFIED MATCH**

5. **NR-005 SFT Final Checkpoint:**
   - Path: `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt`
   - SHA-256: `cbde4b3249cd2f38fa72f64240fa2976a1adb048853a5e7c92c115d03c4662e0`
   - Status: **VERIFIED MATCH**

*(Note: Path clarification: SFT checkpoints reside canonically under `experiments/genesis_conversational_foundation_005/checkpoints/`, committed via Git LFS in commit `fef3410`.)*

---

## 4. Hardware Resource Envelope

The local hardware runtime was probed directly:
- **Processor:** Intel Core i9-13900H (14 cores / 20 threads)
- **Host RAM:** 31.64 GB total (16.28 GB available)
- **Accelerator:** NVIDIA RTX A1000 Laptop GPU
  - Dedicated VRAM: 6.0 GB (6,144 MB)
  - Compute Capability: 8.6 (Ampere)
  - Current Reserved VRAM: 0.0 MB
- **Constraints & Governance:**
  - Maximum safe training batch allocation: $\le 4.5\text{ GB}$ VRAM to prevent Out-Of-Memory (OOM) paging.
  - Mixed precision (`torch.cuda.amp.autocast`) and gradient accumulation mandatory for any model $> 10\text{M}$ parameters.
  - Zero concurrent background GPU training runs permitted.
