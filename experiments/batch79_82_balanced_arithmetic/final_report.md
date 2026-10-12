# ChakrView Master Sprint Batches 79–82: Final Report

## Executive Summary

Sprint Batches 79–82 designed and evaluated a **balanced synthetic arithmetic pretraining curriculum** spanning sums 0 through 18, with strict unordered-pair split isolation and uniform sum-class representation, to test whether eliminating answer-class imbalance induces genuine arithmetic generalization in **ChakrMicro v0.1** (3,443,136 parameters).

---

## 1. Key Scientific & Engineering Findings

1. **Unordered-Pair Benchmark Disjointness**:
   - The 100 single-digit addition problems were grouped into 55 unordered pairs $(\min(a, b), \max(a, b))$.
   - By ensuring that $(a, b)$ and $(b, a)$ always stay in the same split, the benchmark guarantees **zero reversed-pair leakage** across Train (64 pairs), Validation (18 pairs), and Held-Out Test (18 pairs).
2. **Sum-Class Balancing Effect**:
   - In Sprint 75–78, unbalanced data caused global modal collapse to `6` (100% of outputs).
   - In Sprint 79–82, uniform sum-class pretraining broke the global attractor into diversified answers (`18`, `10`, `15`).
   - However, deeper analysis revealed that the model adopted a **piecewise threshold heuristic conditioned almost exclusively on the first operand $a$**:
     * $a = 0 \to \text{predicts } 15$ or $10$.
     * $a \in \{1, 2\} \to \text{predicts } 10$ (100%).
     * $a \ge 3 \to \text{predicts } 18$ (100%).
3. **Generalization Performance**:
   - Held-Out Test exact-match accuracy was **1 / 18 (5.56%)**, occurring on `2 + 8 = 10` due to the first-operand threshold heuristic matching the true sum.
   - For all other held-out pairs (e.g. `1 + 4`, `1 + 8`, `2 + 3`), the model predicted `10` or `18`, failing addition.
   - Symmetry diagnostic revealed that $(1, 4)$ predicted `10` while $(4, 1)$ predicted `18`, confirming operand order sensitivity and lack of functional commutativity.
4. **Format & EOS Mastery**:
   - Template cross-entropy loss dropped to **0.0048**; EOS completion reached **100.0%**. Format adherence was flawless.

---

## 2. Benchmark Metrics Across All Evaluated Conditions

| Condition | Train Exact Match | Val Exact Match | Test Exact Match | Symmetry Match | Template Match | Test Ans Loss | EOS Rate | Modal Distribution |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 10.4075 | 0.0% | Wikitable loops |
| **Condition B (Language Control)** | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 12.2556 | 0.0% | Wikitable headers |
| **Condition C (Balanced Curriculum)** | 2 / 64 (3.1%) | 0 / 18 (0.0%) | **1 / 18 (5.6%)** | 1 / 32 (3.1%) | 0 / 36 (0.0%) | **3.3777** | **100.0%** | `18` (72%), `10` (28%) |
| **Overfit Sanity (3 exs)** | 3 / 64 (4.7%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 12.7520 | 100.0% | `1` (94%) |

---

## 3. Verified Checkpoints and Invariants

- Model Parameters: **3,443,136**
- Fresh Init SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Base Step-10k Checkpoint SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Condition B Checkpoint: [`condition_b_generic_control_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch79_82_balanced_arithmetic/experiments/condition_b_generic_control_step_100.pt) (SHA-256: `7225ddc6d7371dd3127998cd22e3b03e8defad418dd0a18a5fd5455ce98df663`)
- Condition C Checkpoint: [`condition_c_balanced_curriculum_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch79_82_balanced_arithmetic/experiments/condition_c_balanced_curriculum_step_100.pt) (SHA-256: `5213b7ef38dafa096c7c03818ef4da88cc05d8c9f8a208b312da889eae798b58`)
- Overfit Checkpoint: [`overfit_sanity_3ex_step_50.pt`](file:///d:/Project/ChakrView/experiments/batch79_82_balanced_arithmetic/experiments/overfit_sanity_3ex_step_50.pt) (SHA-256: `365fe3b28bda18ae5049363065a6efdfc896e38bbf5e4f4fce6ff6a56b2675d6`)
- Preserved User File: [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) kept untouched and uncommitted.

---

## 4. Hardware Environment
- Intel Core i5-1135G7 @ 2.40GHz, 31.64 GB RAM.
- PyTorch 2.14.0+cpu, Windows 11.
- No CUDA or distributed training; execution was CPU-bounded.
