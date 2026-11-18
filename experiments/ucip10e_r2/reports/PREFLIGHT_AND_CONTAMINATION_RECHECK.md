# Phase A — Forensic Preflight & Contamination Recheck Report

**Date:** 2026-11-18  
**Mission:** UCIP-10E-R2 Clean Benchmark Reconstruction, Evaluator Repair & Baseline Re-evaluation  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Current HEAD:** `a544c8c` (`feat(ucip10e-r1): complete independent benchmark, evaluator and reproducibility validation`)

---

## 1. Checkpoint & Runtime Artifact Verification

All protected baseline checkpoints and the runtime session implementation were re-verified via SHA-256:

| Artifact | File Path | Expected Hash | Actual Hash | Status |
|---|---|---|---|:---:|
| **UCIP-09 Medium** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | **MATCH** |
| **UCIP-09R Medium** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | **MATCH** |
| **UCIP-09S Pilot** | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | **MATCH** |
| **Protected File** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |

---

## 2. Independent Re-computation of UCIP-10E Contamination & Redundancy

We independently re-parsed and audited the contaminated 10E benchmark files (`experiments/ucip10e/datasets/`):
- `benchmark_dev.json` (840 nominal probes): **336 unique prompts**, 504 duplicate instances. 76 prompts contaminated directly from training sets.
- `benchmark_sealed.json` (840 nominal probes): **347 unique prompts**, 493 duplicate instances. **81 prompts contaminated directly from training sets**.
- **Development vs Sealed Overlap:** Exactly **81 identical prompt collisions** exist between `benchmark_dev.json` and `benchmark_sealed.json`.

**Finding:** The R1 discovery is 100% confirmed. The old 10E benchmark must be completely replaced by an independently authored R2 benchmark suite.
