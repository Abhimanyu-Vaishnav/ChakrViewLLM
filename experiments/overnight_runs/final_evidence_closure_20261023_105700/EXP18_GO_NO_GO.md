# Exp18 Authorization Decision — Evidence-Backed Verdict

**Audit Reference**: Task Evaluator v0.1 Final Evidence Closure

**Date**: 2026-10-23

**Final Decision**: **GO**

---

## 1. Decision Rationale

Under the strict decision criteria established for this mission:

- All critical test outcomes are evidenced by executed commands and exit code 0 across 147/147 test cases (TEST_EXECUTION.json).

- All 25 published metrics reconcile bit-for-bit with zero discrepancies against raw frozen outputs (METRIC_RECONCILIATION.csv).

- Core scoring behaviors are verified by unit tests, with unexercised behaviors transparently identified as NOT_TESTED (SCORER_TEST_COVERAGE.csv).

- Dual-timestamped integrity verification confirms 100% hash preservation across all canonical and official dataset files (INTEGRITY_MANIFEST.json).

Therefore, the criteria for authorization are satisfied, yielding the definitive decision: GO.

---

## 2. Documented Limitations for Exp18

Exp18 is authorized with the following explicitly documented constraints:

1. Task Correctness vs Turn Termination: Exp17 demonstrated that clean turn termination (+45.0% EOS on test split, +55.0% on normal chat gate) does NOT equate to task correctness (0.0% scorable accuracy). Exp18 must aim for capability acquisition rather than turn-taking boundary imitation.

2. Evaluator Scorable Coverage: TaskEvaluator v0.1 scores 87.5% of records deterministically (35/40), holding out short explanations as NEEDS_REVIEW. Exp18 must evaluate scorable accuracy separately from unscored items.

3. Scorer Test Coverage Enhancements: Dedicated unit tests should be added in a future PR for arithmetic division, instruction transformation, and factual QA focal entity extraction.

---

## 3. Measurable Acceptance Criteria for Exp18

Exp18 must be governed by TaskEvaluator v0.1 and meet these quantitative acceptance gates:

- Turn Termination: Maintain >= 50.0% clean EOS termination rate on held-out test records.

- Structured Formatting: Achieve >= 20.0% valid RFC 8259 JSON objects on structured formatting test records.

- Scorable Task Accuracy: Achieve >= 10.0% non-zero accuracy across arithmetic, factual QA, and extraction.

- Stutter Suppression: Maintain <= 15.0% average trigram repetition rate on the frozen Normal Chat Gate v0.1 suite.

- Zero Regressions: Zero failures or errors across the 147 repository and evaluator test suite.