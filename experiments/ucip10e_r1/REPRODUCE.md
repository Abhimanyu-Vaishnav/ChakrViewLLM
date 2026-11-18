# UCIP-10E-R1 Reproduction Guide & Exact Execution Commands

This document provides exact, deterministic commands to reproduce every benchmark evaluation, adversarial audit, and test suite in UCIP-10E-R1.

---

## 1. Environment & Setup
Use the existing CPU-first Python virtual environment on Windows:
```powershell
# Verify python environment
.\.venv\Scripts\python.exe --version
```

---

## 2. Benchmark Suite Validation & Registry Check
Validate cryptographic hashes and probe schemas across all benchmark manifests:
```powershell
.\.venv\Scripts\python.exe experiments/ucip10e/runners/eval_engine.py validate
```

---

## 3. Running the Complete 840-Probe Sealed Suite
To execute full-suite evaluations across all three checkpoints:
```powershell
# 1. UCIP-09 Primary Baseline
.\.venv\Scripts\python.exe experiments/ucip10e_r1/scripts/run_full_840.py releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt

# 2. UCIP-09R Degraded Candidate
.\.venv\Scripts\python.exe experiments/ucip10e_r1/scripts/run_full_840.py releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt

# 3. UCIP-09S Condition C Pilot
.\.venv\Scripts\python.exe experiments/ucip10e_r1/scripts/run_full_840.py experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt
```
Results are saved under `experiments/ucip10e_r1/results/full_840/`.

---

## 4. Running the Multi-Checkpoint Regression Comparison
```powershell
.\.venv\Scripts\python.exe experiments/ucip10e/runners/eval_engine.py compare `
  --baseline experiments/ucip10e/results/eval_chakrview_ucip09_medium_model_sealed_core100.json `
  --candidate experiments/ucip10e/results/eval_chakrview_ucip09r_medium_model_sealed_core100.json
```

---

## 5. Running the Multi-Run Reproducibility Audit (30 Sampled Probes)
```powershell
.\.venv\Scripts\python.exe experiments/ucip10e_r1/scripts/run_reproducibility_audit.py
```

---

## 6. Running Adversarial Evaluator and Automated Unit Tests
```powershell
# Run all unit tests (23 passed)
.\.venv\Scripts\pytest.exe experiments/ucip10e_r1/tests/test_ucip10e_r1_comprehensive.py experiments/ucip10e/tests/

# Run the 14 adversarial evaluator tests
.\.venv\Scripts\pytest.exe experiments/ucip10e_r1/tests/test_adversarial_evaluators.py
```
