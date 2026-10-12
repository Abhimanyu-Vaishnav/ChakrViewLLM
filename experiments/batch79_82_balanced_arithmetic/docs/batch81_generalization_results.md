# Batch 81 — True Generalization Evaluation Results

## 1. Experimental Overview

Batch 81 evaluated whether balanced arithmetic curriculum pretraining on single-digit addition (covering sums 0 through 18 with equal sum-class representation) induces true mathematical generalization on genuinely unseen operand pairs.

### Evaluated Conditions:
- **Baseline (Step 10,000 Unmodified)**: Checkpoint `training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Condition B (Matched Generic Language Control)**: 100 steps on 63,794 tokens from Stage C.
- **Condition C (Balanced Arithmetic Curriculum)**: 100 steps on 63,794 tokens from uniform sum-class multi-template addition data (150 documents per sum class 0..18).
- **Overfit Sanity Control**: 50 steps on 3 training examples.
- **Hardware Profile**: CPU-only execution (Intel Core i5-1135G7, 31.64 GB RAM, PyTorch 2.14.0+cpu).

---

## 2. Quantitative Results Across All Evaluated Splits

| Condition / Strategy | TRAIN Exact Match | VAL Exact Match | HELD-OUT TEST Exact Match | Symmetry Diagnostic | Template Robustness | EOS Completion Rate | Mean Test Ans Loss | Modal Collapse Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Step 10k)** | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 0.0% | 10.4075 | 0.0% (Wikitable loops) |
| **Condition B (Language Control)** | 0 / 64 (0.0%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 0.0% | 12.2556 | 100.0% (`|- \n| scope...`) |
| **Condition C (Balanced Curriculum)** | 2 / 64 (3.1%) | 0 / 18 (0.0%) | **1 / 18 (5.6%)*** | 1 / 32 (3.1%)* | 0 / 36 (0.0%) | **100.0%** | **3.3777** | Diversified (`18`, `10`, `15`) |
| **Overfit Sanity Control (3 exs)** | 3 / 64 (4.7%) | 0 / 18 (0.0%) | 0 / 18 (0.0%) | 0 / 32 (0.0%) | 0 / 36 (0.0%) | 100.0% | 12.7520 | 94.4% (`1`) |

*\*Analysis of Held-Out Test "1/18" Match*:
- In Condition C, the only held-out match was `2 + 8 = 10`.
- The model generated `10` because its first-operand heuristic mapped $a \in \{1, 2\}$ to `10`. For `1 + 4`, `1 + 8`, and `2 + 3`, the model also generated `10` (which was incorrect for those pairs).

---

## 3. Detailed Mechanistic Diagnosis: From Global Modal Collapse to Piecewise First-Operand Heuristic

Uniform sum-class balancing successfully broke the global single-digit modal collapse (which collapsed 100% of outputs to `6` in Sprint 75–78). However, inspecting the outputs across all 100 pairs revealed a remarkable structural insight into ChakrMicro's causal attention routing:

### First-Operand Conditioning Table:
Across Train (64 pairs), Validation (18 pairs), and Held-Out Test (18 pairs), the model's predictions partition deterministically based almost entirely on the **first operand $a$**:
- If $a = 0$: predicts `15` (60%) or `10` (40%).
- If $a \in \{1, 2\}$: predicts **`10`** (100% of predictions).
- If $a \in \{3, 4, 5, 6, 7, 8, 9\}$: predicts **`18`** (100% of predictions).

### Scientific Finding:
1. **The Model Conditioned on Input Tokens**: Unlike Sprint 75–78 where the model was blind to input tokens and emitted a constant `6`, the balanced curriculum forced the attention mechanism to attend to the prompt.
2. **Causal Attention Bottleneck**: In the standard autoregressive prompt format `Add: a + b = ?\nassistant\n`, token $a$ appears before token $b$. Instead of composing $a$ and $b$ non-linearly across the 6 attention layers, the model developed a **piecewise threshold heuristic based solely on $a$**:
   * Small digits ($a \le 2$) $\to$ predict `10`.
   * Large digits ($a \ge 3$) $\to$ predict `18`.
3. **Generalization Failure on Held-Out Pairs**: Because the second operand $b$ is largely ignored by the final hidden state at the answer position, the model cannot perform true addition. Out of 18 genuinely held-out test pairs, exact accuracy was 1/18 (5.56%), matching the probability of hitting a sum of 10 by threshold heuristic.

---

## 4. Symmetry & Template Robustness Diagnostics

- **Symmetry Diagnostic (32 off-diagonal pairs)**:
  - For $a \in \{1, 2\}$ and $b \ge 3$, $(a, b)$ yields `10`, whereas the reversed pair $(b, a)$ yields `18`.
  - Example: `1 + 4 = 10`, but `4 + 1 = 18`.
  - This asymmetry confirms that the model is relying on the position of the first operand rather than computing a commutative mathematical relation.
- **Template Robustness Diagnostic (36 alternate phrasing pairs)**:
  - Exact match was **0 / 36 (0.0%)**.
  - Changing prompt phrasing to `What is a + b?` or `Calculate the sum of a and b.` shifted the output distribution towards `18` (86%) and `15` (14%), proving that formatting variations alter the attention routing to operand tokens.

---

## 5. Checkpoint Verification and Cold-Reload Determinism

All experiment artifacts and checkpoints were verified:
- Checkpoint SHA-256 for Condition B: `7225ddc6d7371dd3127998cd22e3b03e8defad418dd0a18a5fd5455ce98df663`
- Checkpoint SHA-256 for Condition C: `5213b7ef38dafa096c7c03818ef4da88cc05d8c9f8a208b312da889eae798b58`
- Checkpoint SHA-256 for Overfit Sanity: `365fe3b28bda18ae5049363065a6efdfc896e38bbf5e4f4fce6ff6a56b2675d6`

Cold reload in an independent evaluator process reproduced exact test matches (1/18) and mean answer-token loss (3.3777) bit-for-bit.
