# Exp18 Go / No-Go Decision & Acceptance Gates

**Audit Reference**: Task Evaluator v0.1 Final Reconciliation

**Date**: 2026-10-23

**Final Decision**: **GO_FOR_EXP18_UNDER_EVALUATOR_GOVERNANCE**

---

## 1. Decision Rationale

The goal of Exp17 was to establish instruction-following and conversation SFT foundations without pretrained external weights.

Exp17 successfully achieved:

1. Turn boundary learning: 45.0% EOS termination on held-out test split, 60.0% on Normal Chat Gate v0.1 (vs 0% and 5% baseline).

2. Severe repetition suppression: Trigram repetition reduced from 46.2% to 13.3%.

3. Offline Deterministic Evaluation Engine: TaskEvaluator v0.1 is verified, tested across 147 unit/regression tests, and prevents false-positive capability claims.

However, task accuracy remains 0.0% across all 8 capabilities due to parameter capacity (3.44M dense) and small synthetic dataset scale (320 training examples).

Proceeding to Exp18 is recommended to address capability acquisition while strictly governed by TaskEvaluator v0.1.

---

## 2. Mandatory Acceptance Criteria for Exp18

To achieve a passing evaluation in Exp18, the model must meet the following quantitative criteria:

1. Turn Termination: Maintain >= 50.0% clean EOS termination rate across held-out test records.

2. Structured Formatting: Achieve >= 20.0% valid RFC 8259 JSON objects on structured formatting test records.

3. Scorable Task Accuracy: Achieve >= 10.0% non-zero scorable accuracy across arithmetic, factual QA, and extraction.

4. Stutter / Repetition Rate: Maintain <= 15.0% average trigram repetition rate on Normal Chat Gate v0.1.

5. Zero Regressions: All 147 core and evaluator unit tests must pass with zero failures and zero errors.