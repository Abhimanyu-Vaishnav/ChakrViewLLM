# UCIP-10E Baseline Results & Multi-Checkpoint Matrix

**Date:** 2026-11-18  
**Mission:** UCIP-10E Intelligence Evaluation Infrastructure  
**Suite Evaluated:** `benchmark_sealed_core100.json` (100 Probes across 10 Domains)  
**Raw Evaluation Results:**
- UCIP-09 Baseline: `experiments/ucip10e/results/eval_chakrview_ucip09_medium_model_sealed_core100.json`
- UCIP-09R Degraded: `experiments/ucip10e/results/eval_chakrview_ucip09r_medium_model_sealed_core100.json`
- UCIP-09S Balanced Pilot: `experiments/ucip10e/results/eval_ucip09s_balanced_pilot_sealed_core100.json`

---

## 1. Multi-Checkpoint Capability Matrix

| Capability Domain | Probes | UCIP-09 Medium (Primary Baseline) | UCIP-09R Medium (Degraded) | UCIP-09S Condition C (Balanced Pilot) | Regression Status (09 vs 09R) |
|---|:---:|:---:|:---:|:---:|:---|
| **1. Language Intelligence** | 10 | **4 / 10 (40.0%)** | 0 / 10 (0.0%) | **4 / 10 (40.0%)** | **Severe Regression in 09R (-40%)**; Replay preserved in 09S |
| **2. Mathematics** | 10 | **1 / 10 (10.0%)** | 0 / 10 (0.0%) | **1 / 10 (10.0%)** | Regression in 09R (-10%); Preserved in 09S |
| **3. Exact Copying** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Parity across all (Architectural limit) |
| **4. Logical Reasoning** | 10 | **2 / 10 (20.0%)** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Regression in 09R (-20%) and 09S (-20%) |
| **5. Generalization** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Parity across all |
| **6. Multi-Turn Reasoning** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Parity across all |
| **7. Self-Correction** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Parity across all |
| **8. Uncertainty & Calibration** | 10 | 4 / 10 (40.0%) | **5 / 10 (50.0%)** | **5 / 10 (50.0%)** | Baseline correctly answers knowns; fails abstention |
| **9. Coding Intelligence** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Parity across all |
| **10. Self-Modeling** | 10 | 2 / 10 (20.0%) | **3 / 10 (30.0%)** | 0 / 10 (0.0%) | Identity recall verified in 09 and 09R |
| **TOTAL ACCURACY** | **100** | **13 / 100 (13.00%)** | **8 / 100 (8.00%)** | **10 / 100 (10.00%)** | **Net Regression of -5.00% in UCIP-09R** |

---

## 2. Key Scientific Observations

1. **UCIP-09 Medium Remains the Undisputed Primary Baseline:**
   - UCIP-09 leads across Language (40%), Math (10%), and Logic (20%), achieving the highest aggregate score (13.0%).
2. **UCIP-09R Suffered Devastating Catastrophic Forgetting:**
   - UCIP-09R scored **0.0% in Language, Mathematics, and Logic**. Its training set skew (88.5% arithmetic) destroyed both language and general reasoning capability while failing to produce generalizable math.
3. **UCIP-09S Balanced Replay Successfully Protected Language:**
   - UCIP-09S matched UCIP-09 in Language (40%) and Mathematics (10%), confirming that balanced replay prevents the collapse suffered by UCIP-09R.
