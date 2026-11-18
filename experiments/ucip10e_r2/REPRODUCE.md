# UCIP-10E-R2 Complete Reproduction Guide

This document provides exact, deterministic commands to reproduce every benchmark evaluation, adversarial repair test, and clean baseline score in UCIP-10E-R2.

---

## 1. Environment Verification
```powershell
.\.venv\Scripts\python.exe --version
```

---

## 2. Re-verifying Zero-Contamination and Benchmark Provenance
```powershell
.\.venv\Scripts\python.exe experiments/ucip10e_r2/scripts/verify_r2_provenance.py
```
*Expected Output:* `training_contamination_count: 0`, `dev_sealed_prompt_overlap_count: 0`.

---

## 3. Running Clean R2 Evaluations Across All Checkpoints (460 Probes Each)
```powershell
# 1. UCIP-09 Primary Baseline
.\.venv\Scripts\python.exe experiments/ucip10e_r2/scripts/run_r2_eval.py releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt uc09

# 2. UCIP-09R Degraded Candidate
.\.venv\Scripts\python.exe experiments/ucip10e_r2/scripts/run_r2_eval.py releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt uc09r

# 3. UCIP-09S Condition C Pilot
.\.venv\Scripts\python.exe experiments/ucip10e_r2/scripts/run_r2_eval.py experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt uc09s
```
Results are saved under `experiments/ucip10e_r2/results/`.

---

## 4. Running Multi-Run Determinism / Reproducibility Audit
```powershell
.\.venv\Scripts\python.exe experiments/ucip10e_r2/scripts/run_r2_reproducibility.py
```

---

## 5. Running Automated Unit Tests
```powershell
# Run all R2 comprehensive tests (9 tests)
.\.venv\Scripts\pytest.exe experiments/ucip10e_r2/tests/

# Run R1 adversarial and infrastructure tests (23 tests)
.\.venv\Scripts\pytest.exe experiments/ucip10e_r1/tests/test_ucip10e_r1_comprehensive.py experiments/ucip10e/tests/
```
