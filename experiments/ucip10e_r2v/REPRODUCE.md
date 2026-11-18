# UCIP-10E-R2V Audit Reproduction Guide

This guide describes how to independently verify and reproduce all results of the UCIP-10E-R2V audit suite.

## 1. Environment & Prerequisites

Execute commands using the project virtual environment:
```powershell
.venv\Scripts\python.exe -m pip install -q pytest
```

## 2. Execute Preflight & Hash Audit

Verify that all checkpoints, benchmark datasets, and protected runtime files match their cryptographic SHA-256 signatures:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/verify_all_hashes.py
```

Expected output:
`_overall_status: ALL_MATCHED`

## 3. Recalculate Benchmark Scores from Raw Records

Recalculate the baseline scores across all 460 probes for UCIP-09, UCIP-09R, and UCIP-09S directly from disk:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/audit_score_recalculation.py
```

Expected accuracy: `10 / 460 = 2.1739%` across all three models.

## 4. Run Logic Passes Forensic Audit

Inspect the 30 reported passes to verify prompt de-duplication and affirmative bias:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/audit_logic_passes.py
```

## 5. Investigate Language Failures

Classify the failure modes across all 75 held-out language probes per model:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/audit_language_failures.py
```

## 6. Audit Evaluator False Negatives

Check whether `StrictEvaluatorsR2` rejected any correct answers:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/audit_false_negatives.py
```

Expected: `0 false negatives`.

## 7. Run 30-Probe Deterministic Reproducibility Audit

Re-evaluate 30 stratified probes per checkpoint live on CPU:
```powershell
.venv\Scripts\python.exe experiments/ucip10e_r2v/scripts/run_reproducibility_30.py
```

Expected: `30 / 30 exact matches` for each of UCIP-09, UCIP-09R, and UCIP-09S.

## 8. Run Automated Test Suite

Execute the comprehensive pytest test suite:
```powershell
.venv\Scripts\python.exe -m pytest experiments/ucip10e_r2v/tests/test_ucip10e_r2v_verification.py -v
```
