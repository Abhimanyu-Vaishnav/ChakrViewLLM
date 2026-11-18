# Phase 0 — Preflight and Lineage Audit Report

**Date:** 2026-11-18  
**Audit Mission:** UCIP-10E-R1 Independent Benchmark, Evaluator & Reproducibility Validation  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Current HEAD:** `6649d18` (`feat(ucip10e): implement intelligence evaluation, measurement and regression infrastructure`)  
**Parent Commits:** `d9316f4` (UCIP-09S completion) $\to$ `32100ac` (UCIP-09R completion)

---

## 1. Cryptographic Artifact Verification

All baseline checkpoints, experiment checkpoints, and protected runtime source files were independently verified via SHA-256:

| Artifact Name | Path | Expected SHA-256 | Actual SHA-256 | Verification Status |
|---|---|---|---|:---:|
| **UCIP-09 Medium Baseline** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | **100% BIT-EXACT MATCH** |
| **UCIP-09R Medium** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | **100% BIT-EXACT MATCH** |
| **UCIP-09S Condition C Pilot** | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | **100% BIT-EXACT MATCH** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **100% BIT-EXACT MATCH** |

---

## 2. Model Architecture & Lineage Topology

Each checkpoint was verified through direct PyTorch state inspection:
- **Base Architecture:** ChakrMedium, 12,932,736 parameters.
  - Width: $d_{\text{model}} = 384$
  - Layers: $n_{\text{layers}} = 7$
  - Attention Heads: $n_{\text{heads}} = 6$
  - FFN Dimension: $d_{\text{ffn}} = 896$
  - Vocabulary Size: 4,096 BPE tokens
  - Context Window: 512 tokens
  - Tied lm_head / token embeddings: `True`

### Lineage Tree:
1. `chakrview_ucip08_medium_model.pt` (`fe6f381a...`)
   - Trained: `train_ucip08.py` on 734 samples.
2. $\downarrow$ `chakrview_ucip09_medium_model.pt` (`5513fefe...`) [Primary Baseline]
   - Trained: `train_ucip09.py` on 3,240 balanced samples (125 steps, LR 2e-5, label smoothing 0.05).
3. $\downarrow$ Fork A: `chakrview_ucip09r_medium_model.pt` (`e1b94322...`)
   - Trained: `train_ucip09r.py` on 885 samples (160 steps, LR 2.5e-5, label smoothing 0.03).
   - Skewed: 88.5% arithmetic, 1.4% English, 1.4% Hindi, 0% instruction following.
4. $\downarrow$ Fork B: `ucip09s_balanced_pilot.pt` (`c99dcc0a...`)
   - Trained: `train_ucip09s_pilot.py` on 1,080 samples (80 steps, LR 1.8e-5, label smoothing 0.05).
   - Balanced replay: 74.7% conversational/multilingual/instruction, 25.3% targeted.

---

## 3. Git Invariants and Environment Status

- **Branch:** `exp/cbr-01-conversational-breakthrough`.
- **Working Tree:** Untracked legacy folders (`ucip_05`, `ucip_06`) preserved without deletion. All new audit artifacts will be created under `experiments/ucip10e_r1/`.
- **No Remote Operations:** Remote push is strictly forbidden.
