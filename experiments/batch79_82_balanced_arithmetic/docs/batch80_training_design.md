# Batch 80 — Balanced Curriculum and Training Design Specification

## 1. Experimental Strategy & Hypotheses

In Batches 75–78, training on an unbalanced synthetic arithmetic dataset caused ChakrMicro to collapse entirely to the constant output `6` (the median/mode of that dataset's sum distribution).

### Core Research Question:
Does eliminating answer-class frequency imbalance prevent modal collapse and induce operand-conditioned addition generalization?

### Experimental Hypotheses:
1. **$H_0$ (Unconditioned Attractor Hypothesis)**: Even when training data presents an exact uniform distribution across all 19 sum classes ($0 \dots 18$), ChakrMicro's causal architecture lacks the routing capacity to condition on $(a, b)$, shifting its output to another attractor or failing held-out generalization ($0\%$ exact match).
2. **$H_1$ (Balanced Learning Alternative)**: By removing the loss advantage of predicting any single modal digit, backpropagation forces attention weights to condition on the operand tokens, producing non-zero, non-constant exact matches on held-out test operand pairs.

---

## 2. Experimental Conditions

All conditions start from the identical verified canonical checkpoint:
[`training_checkpoint_step_10000.pt`](file:///d:/Project/ChakrView/experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt) (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).

### Condition A: Unmodified Baseline
- Zero additional training steps.
- Evaluated as the frozen pretraining reference.

### Condition B: Matched Generic Language Control
- Continued pretraining for 100 steps on **63,794 tokens** extracted from Stage C (`shard_00000.bin`).
- Batch size: 4, Sequence length: 64 ($100 \times 4 \times 64 = 25,600$ trained tokens).
- AdamW optimizer (`lr=5e-4`, `weight_decay=0.01`, `grad_clip=1.0`), seed 42.

### Condition C: Balanced Arithmetic Curriculum
- Continued pretraining for 100 steps on **63,794 tokens** of balanced multi-template synthetic data.
- **Uniform Sum-Class Stratification**: Exactly 150 documents generated for each sum class from 0 to 18 ($19 \times 150 = 2,850$ total documents).
- Drawn exclusively from the 64 allowed training pairs (zero overlap with validation or test).
- Matched 1:1 with Condition B in steps (100), batch size (4), sequence length (64), learning rate ($5 \times 10^{-4}$), and optimizer settings.

### Sanity Check: Overfit Control
- 50 optimization steps on 3 distinct training examples to verify gradient flow and pipeline fitting capacity.

---

## 3. Pre-Registered Decision Criteria

1. **Beating the Constant-Output Baseline**:
   - In this benchmark, predicting the most common training sum (`6`) achieves **0.0%** on validation and **0.0%** on held-out test.
   - Any valid generalization claim requires held-out test exact match $> 0\%$ accompanied by **non-constant answers conditioned on the input operands**.
2. **Answer vs Template Loss Separation**:
   - Answer-token loss must show progressive descent independent of template token learning.
3. **Reproducibility**:
   - Final checkpoints must reproduce identical exact matches and losses under fresh process reload.
