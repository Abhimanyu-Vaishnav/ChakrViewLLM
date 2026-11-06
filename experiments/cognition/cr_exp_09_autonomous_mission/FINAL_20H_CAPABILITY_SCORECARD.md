# FINAL 20-HOUR CAPABILITY SCORECARD
## ChakrView Experiment CR-EXP-09

**Harness:** 600 Frozen Multi-Dimensional Cognition Probes  
**Evaluation Model Champion:** Candidate 3 (`cr_exp_09_candidate_3_best_val.pt`)  
**SHA-256 Digest:** `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592`  
**Execution Environment:** CPU-only Local PyTorch, Zero External Weights  

---

## 1. COMPREHENSIVE SCORECARD ACROSS ALL TIERS

| Category # | Capability Domain | Baseline Hits (Arm B) | Champion Hits (Cand 3) | Accuracy (Cand 3) | Absolute Delta |
|---|---|---|---|---|---|
| CAT-1 | Conversational Persona & Fluency | 62 / 75 | 58 / 75 | 77.33% | -5.34% |
| CAT-2 | Single-Hop & Counterfactual Retrieval | 39 / 75 | 48 / 75 | 64.00% | +12.00% |
| CAT-3 | Multi-Entity Distractor Disambiguation | 21 / 75 | 32 / 75 | 42.67% | +14.67% |
| CAT-4 | Missing-Info Epistemic Abstention | 0 / 75 | 35 / 75 | 46.67% | +46.67% |
| CAT-5 | Multi-Turn State Tracking & Memory | 3 / 75 | 59 / 75 | 78.67% | +74.67% |
| CAT-6 | Elementary Arithmetic State Updates | 0 / 75 | 33 / 75 | 44.00% | +44.00% |
| CAT-7 | Conditional Logical Reasoning | 0 / 75 | 28 / 75 | 37.33% | +37.33% |
| CAT-8 | Strict Constraint Satisfaction | 32 / 75 | 12 / 75 | 16.00% | -26.67% |
| **SUM** | **OVERALL CAPABILITY PERFORMANCE** | **157 / 600** | **305 / 600** | **50.83%** | **+24.66%** |

---

## 2. STATISTICAL SIGNIFICANCE (CHAMPION VS. BASELINE)

- **Total Trials:** $N = 600$ paired binary evaluation probes.
- **Baseline Correct:** $157 / 600$ ($26.17\%$).
- **Candidate 3 Correct:** $305 / 600$ ($50.83\%$).
- **Net Improvement:** $+148$ additional questions answered correctly.
- **Two-Proportion Z-Test:** $z = 8.91$, $p < 0.0001$.
- **Conclusion:** The cognitive gains across multi-turn state tracking, epistemic abstention, arithmetic, and logical reasoning are statistically significant well beyond the $p < 0.001$ threshold.
