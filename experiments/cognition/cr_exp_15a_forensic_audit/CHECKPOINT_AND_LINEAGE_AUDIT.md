# CR-EXP-15A Checkpoint and Lineage Forensic Audit
## Checkpoint Integrity, Parameter Counts, Architecture Configurations, and Lineage Verification

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Benchmark Engineer  
**Date:** November 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Git Repository Lineage and HEAD Verification

- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current HEAD Commit:** `3ff881c373c8f184e53b08f6d0001a03e1dff214`
- **Commit Message:** `CR-EXP-15: investigate architecture capacity and entity binding failure`
- **Protected File Status:** `chakrview/runtime/session.py` is **byte-for-byte identical** to canonical protected hash:
  `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- **Zero Weight Commit Policy:** Complied with. No `.pt` binary model weights exist in Git tree or commit history.

---

## 2. Checkpoint SHA-256 Hash and Parameter Integrity Audit

Each of the seven models evaluated in CR-EXP-15 was loaded using PyTorch with `strict=True` to confirm exact tensor match with zero missing and zero unexpected keys.

| Model Identifier | File Path | SHA-256 Checkpoint Hash | Architecture Topology ($d_{\text{model}} / L / H / d_{\text{head}} / d_{\text{ff}}$) | Parameter Count | Missing Keys | Unexpected Keys | Strict Loading Status |
|---|---|---|---|---|---|---|---|
| **Parent Baseline** | `experiments/cognition/cr_exp_14_entity_binding/checkpoints/baseline_cr12_b/cr_exp_12_candidate_b_best_val_copy.pt` | `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e` | 192 / 6 / 6 / 32 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Candidate A** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/candidate_a/candidate_a_best_val.pt` | `708b2204cf2aea4a816e0ac266b065fcabac90c942137f28f7718bb1079ff5bb` | 192 / 6 / 6 / 32 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Candidate B** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/candidate_b/candidate_b_best_val.pt` | `860fc75828657b97aa4ed3e41c879b78fea03bbead9d3d67bc4a37b5f101b3a8` | 192 / 6 / 6 / 32 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Candidate C** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/candidate_c/candidate_c_best_val.pt` | `e0e817962ef609a56d9969365c8ca12c6b4539b1f2a9f842f6874d8f3491c2dd` | 192 / 6 / 6 / 32 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Candidate D** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/candidate_d/candidate_d_best_val.pt` | `76649964e0d6732a229b77bc54c0afd14a339bef26eb8f06c04865639a0d2989` | 192 / 6 / 6 / 32 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Arch Arm B** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/arch_arm_b_heads3_d64/arch_arm_b_heads3_d64_best_val.pt` | `bfc1f907acd21c8deef67ba4529682f4a410ac63cf7947a13f83aa5ad2e9b57c` | 192 / 6 / 3 / 64 / 512 | 3,443,136 | None (0) | None (0) | **SUCCESS** |
| **Arch Arm C** | `experiments/cognition/cr_exp_15_architecture_binding/checkpoints/arch_arm_c_width256_d64/arch_arm_c_width256_d64_best_val.pt` | `bfc587af1aac0bd95e0726c2802e8a9fea5107b88bc66df05e88a34509418cb8` | 256 / 6 / 4 / 64 / 682 | 5,767,424 | None (0) | None (0) | **SUCCESS** |

---

## 3. Tokenizer and Artifact Lineage

All seven models and evaluations utilized the canonical ChakrView BPE tokenizer:
- **Tokenizer Directory:** `data/experiments/vocab_4096/`
- **Vocabulary Size:** 4,096 tokens
- **Artifact Hashes:**
  - `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `config.json`: `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589`
- **Frozen Benchmark Hash:**
  `experiments/cognition/cr_exp_15_architecture_binding/benchmark_data/benchmark_1200_sealed.json`  
  SHA-256: `71212797975022ad4382d4d660d08125782ae473b85a5081eb2f12d7da5d28d8`

---

## 4. Checkpoint Lineage Verification & Weight Divergence

To verify that models were genuinely trained and did not silently replicate baseline weights:
- **Baseline Parent vs Candidate D:** Verified that 57 out of 57 parameter weight tensors changed ($\Delta > 10^{-6}$ across all layers).
- **Evaluation Mode Audit:** All evaluation loops explicitly called `model.eval()` and wrapped forward passes inside `torch.no_grad()`.
- **Checkpoint Fallback Audit:** Verified that each model evaluated loaded strictly from its assigned dedicated checkpoint file.

