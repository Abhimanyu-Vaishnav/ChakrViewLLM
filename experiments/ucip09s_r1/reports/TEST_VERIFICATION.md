# Stage 9 — Test Verification and Preservation Audit Report

**Date:** 2026-11-18  
**Audit Mission:** UCIP-09S-R1 Independent Evidence Verification & Evaluator Audit  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`

---

## 1. Focused Test Suite Reproduction

We executed the reported 22 focused tests using pytest:
```powershell
.\.venv\Scripts\pytest.exe tests/test_architecture_loader.py tests/test_inference_session.py tests/test_ucip07r_loader.py
```
**Output:**
```text
============================= test session starts =============================
collected 22 items

tests\test_architecture_loader.py ......                                 [ 27%]
tests\test_inference_session.py .......                                  [ 59%]
tests\test_ucip07r_loader.py .........                                   [100%]

============================= 22 passed in 8.60s ==============================
```
**Result:** **22 / 22 PASSED (100% REPRODUCED)**.

---

## 2. Full Repository Test Suite Status

We ran test collection across the entire repository test suite:
```powershell
.\.venv\Scripts\pytest.exe --collect-only tests/
```
**Output:**
```text
======================== 3165 tests collected in 3.94s ========================
```
- **Total Repository Tests:** 3,165 test cases collected across 95 test modules.
- **Scope Verification:** The UCIP-09S mission reported only the **22 focused regression tests** for runtime architecture loading, tied weights, and inference sessions. The full repository test suite (3,165 tests covering past batches and legacy components) was **not run in UCIP-09S** due to duration constraints.
- **Reporting Separation:**
  - Focused Tests Run: 22 / 22 Passed.
  - Full Test Suite Collected: 3,165 tests.
  - Tests Not Run in UCIP-09S: 3,143 tests.

---

## 3. Protected File & Canonical Checkpoint Hashes

### Protected File Integrity:
- File: `chakrview/runtime/session.py`
- Protected Hash: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- Verified Actual Hash: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- Status: **100% BIT-EXACT MATCH (PROTECTED & UNTOUCHED)**.

### Canonical and Baseline Checkpoints:
| Checkpoint | File Path | Verified SHA-256 |
|---|---|---|
| **UCIP-08 Medium** | `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt` | `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631` |
| **UCIP-09 Medium Baseline** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` |
| **UCIP-09R Medium** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` |
| **UCIP-09S Condition C Pilot** | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` |

All hashes match existing manifests. Zero checkpoint files were modified or corrupted.
