# Batch 124 — Isolated, Deterministic Two-Digit Benchmark Dataset

## Executive Summary

Batch 124 designed and generated a reproducible, isolated dataset for calculator addition using two-digit nonnegative operands from 10 through 99.

The dataset addresses two core failure modes identified in previous sprints:
1. **Syntactic Fragility**: Previously, models trained on a single fixed template failed to generalize when sentence structure changed. This benchmark introduces **8 distinct training prompt templates** and reserves **2 held-out prompt templates exclusively for evaluation**.
2. **Operand Generalization Beyond Single Digits**: Evaluates numbers in the range $[10, 99]$ to determine whether operand binding learned on character/digit tokens scales to multi-token and multi-digit representations.

---

## 1. Dataset Architecture & Proportions

The dataset was generated deterministically using seed `42` via [`experiments/batch123_126_prompt_diverse_multidigit/data/generate_multidigit_dataset.py`](file:///d:/Project/ChakrView/experiments/batch123_126_prompt_diverse_multidigit/data/generate_multidigit_dataset.py).

### Split Counts and Proportions:
- **Training Set (`train.jsonl`)**: 100 records
  - Tool Invocation Examples: 70 records (70.0%) across 60 unique two-digit pairs and 8 prompt templates.
  - Negative Control Prompts: 30 records (30.0%) containing general knowledge, physics, coding, and underspecified math prompts with missing operands.
  - Exact proportion: **70% tool / 30% control**.
- **Validation Set (`val.jsonl`)**: 20 records
  - Tool Invocation Examples: 15 records (75.0%)
  - Negative Controls: 5 records (25.0%)
- **Test Set (`test.jsonl`)**: 45 records
  - Partition 1 (`familiar_pairs_unseen_templates`): 10 examples (22.2%)
    - Familiar two-digit pairs combined with held-out prompt templates.
  - Partition 2 (`unseen_pairs_familiar_templates`): 15 examples (33.3%)
    - Strictly unseen two-digit pairs combined with familiar prompt templates.
  - Partition 3 (`unseen_pairs_unseen_templates`): 10 examples (22.2%)
    - Strictly unseen two-digit pairs combined with held-out prompt templates (double generalization).
  - Partition 4 (`test_negative_control`): 10 examples (22.2%)
    - Fresh negative controls and underspecified addition prompts.

---

## 2. Template Definitions & Isolation

### Training Templates (8 distinct phrasings):
1. `Calculate sum of:\nA={a}\nB={b}`
2. `Please add A={a} and B={b}.`
3. `Compute addition for A={a} and B={b}.`
4. `Find the total sum of A={a} and B={b}.`
5. `Sum operands: A={a}, B={b}.`
6. `Perform add operation on A={a} and B={b}.`
7. `Calculate total: A={a} + B={b}.`
8. `What is the sum of A={a} and B={b}?`

### Held-Out Evaluation Templates (2 strictly reserved phrasings):
1. `Execute calculator addition: first={a}, second={b}.`
2. `Determine arithmetic total for inputs: A is {a} and B is {b}.`

**Isolation Verification**: Neither held-out template nor their syntactic key patterns occur in any training or validation record.

---

## 3. Overlap & Leakage Audit

Automated overlap audit verified:
1. **Direct Pair Overlap**: Unseen test pairs in Partition 2 and Partition 3 have **0 overlap** with training pairs.
2. **Reversed Pair Overlap**: Unseen test pairs have **0 overlap** with reversed training pairs ($b, a$).
3. **Template Overlap**: Partitions 1 and 3 use exclusively the two held-out templates.
4. **Reproducibility**: All files generated with seed `42` and hashed with SHA-256 in [`experiments/batch123_126_prompt_diverse_multidigit/data/manifest.json`](file:///d:/Project/ChakrView/experiments/batch123_126_prompt_diverse_multidigit/data/manifest.json).
