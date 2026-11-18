# UCIP-10E-R2V Audit 1: Preflight Verification & Hash Integrity

**Mission ID:** UCIP-10E-R2V  
**Audit Phase:** Baseline Integrity & Hash Verification  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**HEAD Commit:** `1afaf15` (`feat(ucip10e-r2): complete clean benchmark reconstruction, evaluator repair and baseline re-evaluation`)  

---

## 1. Executive Summary

This preflight audit independently verifies that all repository artifacts, checkpoints, clean benchmark datasets, manifests, and protected runtime components remain completely unmodified and bit-exact across the repository.

All SHA-256 hashes were calculated directly from disk bytes using Python 3.14.7 standard cryptographic libraries (`hashlib.sha256`) and compared against mission-critical specifications.

---

## 2. Cryptographic Hash Verification Table

| Component | Disk Path | Observed SHA-256 | Expected SHA-256 | Status |
|---|---|---|---|---|
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (BIT-EXACT)** |
| **UCIP-09 Medium** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | **PASS (BIT-EXACT)** |
| **UCIP-09R Medium** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | **PASS (BIT-EXACT)** |
| **UCIP-09S Condition C** | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | **PASS (BIT-EXACT)** |
| **Sealed Benchmark R2** | `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` | `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` | `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` | **PASS (BIT-EXACT)** |
| **Dev Benchmark R2** | `experiments/ucip10e_r2/datasets/development/benchmark_r2_development.json` | `29b213badefba55bd76cece47afb6f59d694cfe2040037970a3384539e610b3c` | `29b213badefba55bd76cece47afb6f59d694cfe2040037970a3384539e610b3c` | **PASS (BIT-EXACT)** |
| **Manifest Registry R2**| `experiments/ucip10e_r2/manifests/manifest_registry_r2.json` | `f44c5af7f340e4d8873da01cf08d953e0af4e6503d221d5b56eb64870943b49a` | `f44c5af7f340e4d8873da01cf08d953e0af4e6503d221d5b56eb64870943b49a` | **PASS (BIT-EXACT)** |
| **Strict Evaluators R2**| `experiments/ucip10e_r2/evaluators/strict_evaluators_r2.py` | `332e701c293dabfc40d0403143458a69a3022ecf10738e8f54bb0ac5217ddac4` | Recorded baseline | **PASS (AUTHENTIC)** |

---

## 3. Boundary & Constraint Audit

1. **Zero Checkpoint Mutation:** No checkpoint files have been touched, moved, rewritten, or trained.
2. **Zero Protected Code Modification:** `chakrview/runtime/session.py` matches the inviolable root hash `9080ff...`.
3. **Zero R2 File Modification:** All raw evaluation outputs (`experiments/ucip10e_r2/results/*`), benchmark datasets, and evaluator scripts are in read-only audit mode.
4. **Git Branch Isolation:** The working directory is on `exp/cbr-01-conversational-breakthrough`. All audit artifacts are isolated exclusively under `experiments/ucip10e_r2v/`.
