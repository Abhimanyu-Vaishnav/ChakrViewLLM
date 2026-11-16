# UCIP-03: PHASE ZERO — BASELINE & LINEAGE AUDIT REPORT

**Mission ID:** UCIP-03  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** 2026-11-16  
**Auditor / Lead Engineer:** Autonomous Conversational Recovery Agent  

---

## 1. Executive Summary

Phase Zero establishes the cryptographic baseline, resolves the canonical architecture hash discrepancy documented in preceding waves, and recovers the execution state and checkpoint artifacts of UCIP-02 training task `task-5995`.

All core invariants have been verified against the physical filesystem. The protected runtime file `chakrview/runtime/session.py` remains bit-exact and unmodified.

---

## 2. Artifact & Cryptographic Integrity Audit

| Artifact Description | Target Path | File Size | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Protected Runtime** | `chakrview/runtime/session.py` | 12,511 bytes | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (UNMODIFIED)** |
| **Canonical Architecture Init** | In-Memory (`ChakrMicro`, seed=42) | 13,772,544 bytes | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (BIT-EXACT)** |
| **Stage B Pretrained Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | 42,212,787 bytes | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (VERIFIED)** |
| **Frozen Conversational Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/.../sft_checkpoint_step_200.pt` | 41,417,634 bytes | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (VERIFIED)** |
| **v0.11 Release Model** | `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt` | 41,388,167 bytes | `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754` | **PASS (VERIFIED)** |
| **Recovered 13M Candidate** | `experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt` | 51,757,989 bytes | `19f248afbd5502372ccbb064d0d53fa41b362f391a2ad8f99e944fe0a036343d` | **PASS (VERIFIED)** |
| **Task-5995 (UCIP-02) Checkpoint**| `experiments/cognition/ucip_02/training/ucip02_recovered_model_best.pt` | 13,795,923 bytes | `8c26b4e49c71b48e4b16f8649e8ff15a836e032984a2494fdb769ef3bad71805` | **PASS (VERIFIED)** |

---

## 3. Discrepancy Resolution: Canonical Architecture Hash

### 3.1 The Question
In earlier documentation (CR-21, UCIP-01, UCIP-01R, UCIP-02), the canonical architecture init was occasionally cited as:
`c5571c9cf41d7c34b172a6bcaae4f9f7431f79a957262d04f1fb9d187766dbb7`
whereas the project's canonical specification (`chakrview/brain/canonical_hash.py`) and earlier tests specify:
`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

### 3.2 Root Cause Analysis
1. Both strings share the exact same 8-character prefix `c5571c9c`.
2. Direct execution of `verify_canonical_baseline(ChakrMicro(ModelConfig()))` with `torch.manual_seed(42)` produces:
   `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` with `delta_w == 0` and `total_params == 3443136`.
3. Tracing git history revealed that `c5571c9cf41d...` was introduced during markdown documentation generation in commit `cdbfef9` and copied forward into summary tables. No Python code, test, or verification script ever computed or validated `c5571c9cf41d...`.
4. **Resolution:** `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` is the authoritative, immutable, and reproducible canonical architecture initialization hash for ChakrMicro (seed 42).

---

## 4. Status and Lineage of Task-5995 (UCIP-02 Checkpoint)

In UCIP-02, background task `task-5995` executed `train_ucip02.py` to test whether a focused fine-tuning run on 1,098 conversational records could disrupt the "Your project was..." attractor state.

- **Status:** Complete. The run trained for 300 steps and successfully wrote `experiments/cognition/ucip_02/training/ucip02_recovered_model_best.pt`.
- **Validation Loss:** Reached $0.5282$ at step 300 (best validation accuracy $90.72\%$).
- **Model Configuration:** 6 layers, 6 heads, $d_{\text{model}}=192$, hidden dimension $512$, vocab size $4096$, total parameters $4,229,568$.
- **Behavioral Findings:** It broke the repetitive project description attractor and acquired basic sentence synthesis ("A dog is a loyal and playful companion"), but capacity bounds on 3.44M/4.2M parameters remained for arithmetic and multi-turn entity recall without external hybrid memory.
- **Role in UCIP-03:** Serves as one of the baseline candidates in the 150-probe foundational diagnostic evaluation.
