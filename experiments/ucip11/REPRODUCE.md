# UCIP-11 Reproduction Guide

This document describes how to independently verify, train, and evaluate all results from the UCIP-11 research mission.

## 1. Environment & Setup

Ensure the virtual environment is activated:
```powershell
.venv\Scripts\python.exe -m pip install -q pytest psutil
```

## 2. Execute Preflight & Corpus Audits

Run the preflight checks and dataset audits:
```powershell
.venv\Scripts\python.exe experiments/ucip11/scripts/audit_corpora.py
.venv\Scripts\python.exe experiments/ucip11/scripts/audit_tokenizer.py
```

## 3. Verify Training Pipeline Correctness

Validate causal shift alignment, loss masking, and overfitting sanity:
```powershell
.venv\Scripts\python.exe experiments/ucip11/training/pipeline_validation.py
```
Expected output: `Pipeline alignment and gradient flow tests PASSED.` and `Tiny overfitting sanity test PASSED.`

## 4. Build Independent Benchmark Suites

Generate and freeze the development and sealed suites:
```powershell
.venv\Scripts\python.exe experiments/ucip11/scripts/build_ucip11_benchmarks.py
```

## 5. Execute Controlled Training Runs

Train Conditions B (CPT), C (SFT), D (CPT+SFT), and E (Scratch):
```powershell
.venv\Scripts\python.exe experiments/ucip11/training/train_all_conditions.py
```

Checkpoints will be saved under `experiments/ucip11/checkpoints/`.

## 6. Run Benchmark Evaluations

Evaluate all conditions on the 100-probe development and 100-probe sealed suites:
```powershell
.venv\Scripts\python.exe experiments/ucip11/scripts/run_benchmark_evaluations.py
```

Results are saved to `experiments/ucip11/results/full_benchmark_evaluation_results.json`.

## 7. Run Automated Test Suite

Execute the full suite of verification tests:
```powershell
.venv\Scripts\python.exe -m pytest experiments/ucip11/tests/test_ucip11_verification.py -v
```
Expected: `7 passed`.
