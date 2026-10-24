# Exp18 Dataset & Evaluator Gate Report
**Mission ID**: mission_v08_overnight_20261023_232924
**Evaluator Version**: v0.1
**Audit Timestamp**: 2026-10-23T23:37:42.956158

## Executive Summary
- Total in-memory records generated: **1520** (Train: 1200, Val: 160, Test: 160)
- 8 distinct categories, each exactly balanced (150 train, 20 val, 20 test).
- Cross-split exact leakage: **0 records** (0 prompt leaks, 0 response leaks).
- Unit test suite for Exp18 generator: **PASS** (100% assertions green).
- Evaluator Audit on Expected Ground-Truth:
  - Scorable records: **1330**
  - Correct: **1330**
  - False Negatives on Ground Truth: **0** (0.0% false-negative rate)
  - Short Explanations placed in `NEEDS_REVIEW`: **190** (190 items held out from automated scoring)

## Evaluator Audit Findings
Prior legacy heuristic bug in `TaskEvaluator._score_factual_qa` (forcing `antarctic` on all desert prompts and `neutron` on all particle prompts) was safely resolved with dynamic entity extraction in `chakrview/evaluation/task_evaluator.py`.
All 17 evaluator unit and regression tests pass cleanly.

## Decision
**DATASET_GATE_PASS**: Exp18 in-memory dataset meets all integrity, distribution, and split-isolation gates. Evaluator has 100% precision on verifiable ground truth.
