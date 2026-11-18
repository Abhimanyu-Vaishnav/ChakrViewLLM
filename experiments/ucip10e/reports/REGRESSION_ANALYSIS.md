# UCIP-10E Regression and Tradeoff Analysis

**Date:** 2026-11-18  
**Mission:** UCIP-10E Intelligence Evaluation Infrastructure  

---

## 1. Regression Comparison Summary

Using the automated CLI comparison tool:
`python experiments/ucip10e/runners/eval_engine.py compare --baseline <09_eval> --candidate <09r_eval>`

```text
==========================================================================
UCIP-10E CAPABILITY COMPARISON & REGRESSION AUDIT
Baseline:  chakrview_ucip09_medium_model.pt (Acc: 13.00%)
Candidate: chakrview_ucip09r_medium_model.pt (Acc: 8.00%)
Overall Delta: -5.00%
==========================================================================
Domain                 | Baseline     | Candidate    | Delta    | Status
--------------------------------------------------------------------------
coding                 |   0.0% (0/10) |   0.0% (0/10) |   +0.0% | PARITY
copying                |   0.0% (0/10) |   0.0% (0/10) |   +0.0% | PARITY
generalization         |   0.0% (0/10) |   0.0% (0/10) |   +0.0% | PARITY
language               |  40.0% (4/10) |   0.0% (0/10) |  -40.0% | REGRESSION
logic                  |  20.0% (2/10) |   0.0% (0/10) |  -20.0% | REGRESSION
mathematics            |  10.0% (1/10) |   0.0% (0/10) |  -10.0% | REGRESSION
multi_turn             |   0.0% (0/10) |   0.0% (0/10) |   +0.0% | PARITY
self_correction        |   0.0% (0/10) |   0.0% (0/10) |   +0.0% | PARITY
self_modeling          |  20.0% (2/10) |  30.0% (3/10) |  +10.0% | IMPROVED
uncertainty            |  40.0% (4/10) |  50.0% (5/10) |  +10.0% | IMPROVED
--------------------------------------------------------------------------
Regression Verdict: CRITICAL REGRESSIONS DETECTED
```

---

## 2. Capability Tradeoffs & Gate Enforcement

1. **Promotion Rejection:** Under strict UCIP-10E gate rules, **UCIP-09R cannot be promoted**. It introduced severe regressions in three core foundational domains (Language $-40\%$, Logic $-20\%$, Mathematics $-10\%$).
2. **Tradeoff Analysis:** Fine-tuning on a small, narrow task distribution without anchor replay destroys general capabilities. Even if a model slightly alters token frequencies on one task, the cost of wiping out conversational language is catastrophic.
