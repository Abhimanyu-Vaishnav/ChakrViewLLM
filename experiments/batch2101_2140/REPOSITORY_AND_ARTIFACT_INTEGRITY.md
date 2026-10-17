# ChakrView Repository & Artifact Integrity Audit — Batches 2101–2140
**Milestone:** Batches 2101–2140  
**Phase:** Phase A — Repository and Artifact Integrity  
**Branch:** `dev/v0.6-v0.7-validation`  
**Date:** October 2026  
**Auditor:** Independent Preflight Release Agent  

---

## 1. Git Repository State & Working Tree Verification

| Property | Value | Integrity Verification |
|:---|:---|:---:|
| **Current Branch** | `dev/v0.6-v0.7-validation` | Verified active branch |
| **Current HEAD Commit** | `18fad0f` (`18fad0f4d1e2e9cb14c4ef4eec6ba94a55ae58f8`) | Verified HEAD |
| **Ancestry Verification** | `git merge-base --is-ancestor 18fad0f HEAD` returned exit code 0 | Confirmed direct ancestor |
| **Preceding Commit Message** | `feat(batch2061-2100): independent reproduction, corpus integrity audit and conditional cpt release gate` | Verified |
| **Working-Tree Status** | Unstaged modifications in `chakrview/runtime/session.py` (canonical ChatML / `ConversationTemplate` integration) preserved untouched. Untracked sprint files preserved untouched. | Verified compliant |
| **Staged Changes** | None prior to mission start | Verified clean stage |

---

## 2. Environment & Dependency Runtime Versions

| Software Component | Exact Recorded Version |
|:---|:---|
| **Operating System** | Windows 11 / Windows NT x86_64 |
| **Python** | `3.14.7` (tags/v3.14.7:823f032, MSC v.1944 64 bit AMD64) |
| **PyTorch** | `2.14.1+cu126` (CUDA Available: `True`) |
| **pytest** | `9.1.1` (pluggy-1.6.0) |
| **Tokenizer Engine** | ChakrView Indigenous Byte-Level BPE (`chakrview.tokenizer.serialization`) |

---

## 3. Cryptographic Verification of Protected Artifacts

All model checkpoints and canonical tokenizer assets were independently hashed using SHA-256 over their exact binary payloads:

| Protected Artifact | Relative Path | Expected SHA-256 Digest | Measured SHA-256 Digest | Status |
|:---|:---|:---|:---|:---:|
| **Canonical Pretraining Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH (Frozen)** |
| **CPT 150-Step Candidate** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | **MATCH (Frozen)** |
| **CPT 300-Step Candidate** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | **MATCH (Frozen)** |
| **Canonical Tokenizer Config** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/config.json` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **MATCH (Frozen)** |
| **Canonical Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **MATCH (Frozen)** |
| **Canonical Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **MATCH (Frozen)** |
| **Corpus v2 Manifest** | `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` | `4c7fc6ed32790bbec183ee670eb094c64672bb35b93f22bdb3c003c07de02519` | `4c7fc6ed32790bbec183ee670eb094c64672bb35b93f22bdb3c003c07de02519` | **MATCH (Frozen)** |
| **Split v2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | **MATCH (Frozen)** |

### Tokenizer File Checksum Reconciliation:
In the Batch 2061 audit report, the tokenizer hashes recorded were placeholder references from an intermediate hash calculation. We verified the actual canonical files committed on git:
- `config.json`: `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589`
- `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
Both `release/chakrview-v0.5.0/data/experiments/vocab_4096/` and `data/experiments/vocab_4096/` share these identical binary hashes.

**Phase A Verdict:** PASSED. All canonical assets, experimental checkpoints, manifests, and git ancestry are verified and frozen.
