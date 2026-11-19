# UCIP-12R: UCIP-11 Evaluation Score Forensic Audit Report

**Mission ID:** UCIP-12R — Forensic Verification  
**Audit Target:** Evaluation records in `experiments/ucip11/results/full_benchmark_evaluation_results.json`  
**Execution Script:** `experiments/ucip12r/scripts/verify_ucip11_scores.py`  

---

## 1. Executive Summary

Phase I conducted an independent forensic audit into whether the 1% evaluation scores reported for UCIP-11 model checkpoints were valid capabilities or false positives caused by evaluator defects.

### Audit Result: **VERIFIED FALSE POSITIVE — TRUE CAPABILITY 0.00%**
All 4 models reporting a 1.00% pass score (Condition A Baseline, Condition B CPT, Condition C SFT, Condition D CPT+SFT) passed **the exact same single probe** (`SEA_INS_001`).
Detailed inspection of raw model outputs and evaluator logs confirmed that **the outputs were incoherent word salad**, yet received a score of $1.0$ because the evaluator applied a length-only heuristic check for the formatting category without verifying semantic correctness.

---

## 2. Forensic Probe Analysis: Probe `SEA_INS_001`

- **Benchmark Prompt:**
  > *"Provide the chemical formulas for water, carbon dioxide, and methane separated by commas only."*
- **Expected Correct Answer:**
  > `"H2O, CO2, CH4"`

### Raw Model Outputs by Condition

| Condition | Raw Generated Output | Evaluator Reason | Reported Score | True Assessment |
|---|---|---|---|---|
| **Condition A (Baseline)** | `"Greenery is the saw gy of an objects toward one mosphere most neural converslower time."` | Length check | 1.0 | **INCOHERENT WORD SALAD (FALSE POSITIVE)** |
| **Condition B (CPT)** | `"Greenerating boower, and werated by werating ars thrst neural converson distant in the interaction of the internorgant ds, and con"` | Length check | 1.0 | **INCOHERENT WORD SALAD (FALSE POSITIVE)** |
| **Condition C (SFT)** | `"The while the reenerather which in the recelecting the where motion of a containd, whil refud, motion to re"` | Length check | 1.0 | **INCOHERENT WORD SALAD (FALSE POSITIVE)** |
| **Condition D (CPT+SFT)** | `"The the while the where the reen parthone the was morefue, which in the the the morece, pator for the the w."` | Length check | 1.0 | **INCOHERENT WORD SALAD (FALSE POSITIVE)** |
| **Condition E (Scratch)** | `""` (Empty / repetition collapse) | Failed | 0.0 | **COLLAPSED (TRUE NEGATIVE)** |

---

## 3. Evaluator Defect Mechanism

In UCIP-11's automated scoring script, probes categorized as `subdomain: formatting` evaluated outputs solely on whether the character length fell below an arbitrary ceiling (e.g. 150 characters). The evaluator did not check for the presence of the required chemical formulas (`H2O`, `CO2`, `CH4`). Because the degenerated models emitted strings under 150 characters before stopping, the evaluator awarded full credit ($1.0$).

---

## 4. Recalculated Benchmark Scores

When evaluated with strict semantic correctness:

| Condition | Checkpoint | Nominal Score | Strict Verified Score |
|---|---|---|---|
| Condition A | UCIP-09 Medium Baseline | 1.00% (1/100) | **0.00% (0/100)** |
| Condition B | Stage B Pretraining (CPT) | 1.00% (1/100) | **0.00% (0/100)** |
| Condition C | Exp34 v3 Instruction Tuning (SFT) | 1.00% (1/100) | **0.00% (0/100)** |
| Condition D | Staged CPT + SFT | 1.00% (1/100) | **0.00% (0/100)** |
| Condition E | Scratch Initialization | 0.00% (0/100) | **0.00% (0/100)** |

**Conclusion:** The claim in UCIP-12 that the historical 1% scores were false positives is **100% VERIFIED**. The historical models exhibited zero functional generalization on unseen sealed probes.
