# UCIP-10E-R2V Audit 2: Score Recalculation Audit

**Mission ID:** UCIP-10E-R2V  
**Audit Objective:** Independently recalculate all three checkpoint baseline scores directly from raw probe evaluation records (`evaluations` array in each JSON) and verify mathematical accuracy against reported summary figures.

---

## 1. Executive Summary

We performed an automated, bit-exact audit of all individual probe evaluations in:
- `experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json`
- `experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json`
- `experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json`

**Finding:** All summary figures reported in the R2 baseline reports match the sum of individual probe evaluation records **exactly 100%**. There is zero mathematical inflation, truncation, or reporting discrepancy.

---

## 2. Recalculated Checkpoint Performance Breakdown

### A. Overall Checkpoint Accuracy

| Checkpoint | Checkpoint SHA-256 | Total Probes Audited | Recalculated Passes | Recalculated Fails | Recalculated Accuracy | Reported Accuracy | Discrepancy |
|---|---|---|---|---|---|---|---|
| **UCIP-09 Medium** | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | 460 | 10 | 450 | **2.17%** (2.1739%) | 2.17% | **0.00%** |
| **UCIP-09R Medium** | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | 460 | 10 | 450 | **2.17%** (2.1739%) | 2.17% | **0.00%** |
| **UCIP-09S Condition C** | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | 460 | 10 | 450 | **2.17%** (2.1739%) | 2.17% | **0.00%** |

---

## 3. Recalculated Domain-Level Breakdown

The exact tally across all 10 evaluated capability domains is identical for all three checkpoints:

| Capability Domain | Total Probes | UCIP-09 Passes | UCIP-09R Passes | UCIP-09S Passes | Domain Accuracy (%) |
|---|---|---|---|---|---|
| **Language Generation** | 75 | 0 | 0 | 0 | **0.00%** |
| **Arithmetic & Math** | 75 | 0 | 0 | 0 | **0.00%** |
| **Copying & Verbatim** | 50 | 0 | 0 | 0 | **0.00%** |
| **Deductive Logic** | 50 | **10** | **10** | **10** | **20.00%** |
| **Generalization & Transfer**| 60 | 0 | 0 | 0 | **0.00%** |
| **Multi-turn Dialogue** | 40 | 0 | 0 | 0 | **0.00%** |
| **Self-Correction** | 30 | 0 | 0 | 0 | **0.00%** |
| **Uncertainty Calibration** | 30 | 0 | 0 | 0 | **0.00%** |
| **Coding & Syntax** | 30 | 0 | 0 | 0 | **0.00%** |
| **Runtime & System Awareness**| 20 | 0 | 0 | 0 | **0.00%** |
| **Total Benchmark** | **460** | **10** | **10** | **10** | **2.17%** |

---

## 4. Verification Conclusion

The R2 scoring logic is strictly verifiable. No phantom points or synthetic adjustments occurred. The 2.17% capability score represents the exact empirical performance of each model when evaluated on the clean, unmemorized sealed benchmark suite.
