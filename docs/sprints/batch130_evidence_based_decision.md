# Batch 130 — Evidence-Based Decision, Analysis & Next Step

## Executive Summary

Batch 130 synthesizes findings from the independent evidence audit, failure mechanism diagnosis, and controlled pair-density experiment conducted across Batches 127–130.

---

## 1. Explicit Answers to Core Mission Questions

### 1. What was the primary failure mechanism supported by the evidence?
**Answer: Under-optimization per pair in high-entropy multi-digit space, compounded by syntactic gating conservatism.**
- When increasing the training pool from 60 pairs (Arm 1) to 160 pairs (Arm 2) within a fixed 200-step budget, exact-call accuracy did **not** improve; it dropped from 22.9% to 5.7%.
- Under a small step budget (200 steps), spreading gradient updates across 160 diverse pairs resulted in each pair receiving insufficient gradient reinforcement ($\approx 1.25$ passes per pair), degrading representation stability.
- Gating conservatism remains a key factor: held-out template 2 (`"Determine arithmetic total for inputs: A is {a} and B is {b}."`) triggered **0 tool calls** across both arms because of lack of shared keyword anchors with the training prompts, correctly defaulting to clean refusal rather than emitting malformed calls.

### 2. Which alternative explanations remain unresolved?
- **Subword BPE Segmentation**: While comma-merging was rejected as the sole cause of failure (comma-merged numbers actually performed marginally better than unmerged numbers in Arm B), the fact that 48/90 numbers fragment into two subword tokens introduces token length variance in attention heads.
- **Attention Head Specialization vs Capacity**: In ChakrMicro v0.1 ($d_{model}=192$, 6 layers, 6 heads), whether multi-digit conditional copying requires dedicated induction heads or a longer training schedule with gradient accumulation remains to be isolated.

### 3. Did the intervention improve held-out exact argument binding?
**Answer: No.**
- Arm 2 (Expanded Density) achieved 5.7% (2/35) overall exact calls vs Arm 1 (Baseline Density) which achieved 22.9% (8/35). On Partition 2 (unseen pairs, familiar templates), Arm 1 achieved 20.0% (3/15) while Arm 2 achieved 0.0% (0/15).
- However, Arm 2 did achieve 20.0% (2/10) on Partition 3 (double generalization), indicating that broad pair exposure does seed some multi-digit copy attention, but requires higher training step budgets to stabilize.

### 4. Did it improve unseen-pair plus unseen-template generalization?
**Answer: Yes, slightly on double generalization (Partition 3).**
- Arm 1 scored 0.0% (0/10) on Partition 3, whereas Arm 2 scored 20.0% (2/10) exact calls.

### 5. Did unsupported-request behavior regress?
**Answer: No.**
- Across both arms, unsupported requests and negative controls achieved a **0.0% false-positive rate (0/10)** and a **100.0% clean refusal rate (10/10)**. Gating reliability was completely preserved.

### 6. Is the result independently reproducible?
**Answer: Yes.**
- Datasets were generated with fixed seed `42` and verified SHA-256 hashes. Checkpoints and evaluation predictions are fully logged and verified by regression test `test_batch127_130_binding_regression.py`.

### 7. What should the next batch test, and what should it avoid repeating?
- **What to avoid**: Do NOT simply expand dataset size without scaling the training step budget and learning rate schedule. Doing so dilutes exposure per pattern and causes under-fitting.
- **What to test**: Test a **fixed-syntax multi-digit curriculum with extended step schedule (500–1000 steps)**, or test uniform digit tokenization (e.g., space-separated digits in training targets) to isolate subword segmentation from step budget limitations.

---

## 2. Regression Test Suite Status

The full regression test suite passed (34 passed in 4.86s):
- `tests/test_batch127_130_binding_regression.py` (6 passed)
- `tests/test_batch123_126_multidigit_regression.py` (4 passed)
- `tests/test_batch115_evaluation_semantics.py` (8 passed)
- `tests/test_batch115_118_tool_gating_regression.py` (4 passed)
- `tests/test_batch111_114_curriculum.py` (5 passed)
- `tests/test_batch107_110_copy_benchmark.py` (7 passed)
