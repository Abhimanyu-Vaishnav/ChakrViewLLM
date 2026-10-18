# Batch 107 — Experiment Integrity and Code Audit

**Sprint**: ChakrView Master Sprint Batches 107–110  
**Module**: Experiment Integrity Audit, Operand Copy Benchmark, and Architecture Decision  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Repository State & Invariant Verification

Before performing audit experiments, the repository state was checked:
- **Git Branch**: `dev/v0.6-v0.7-validation`
- **Current HEAD**: `b4e0189a9f1ccf4466137db11d9647ec8d473d04` (Commit verified as HEAD).
- **Working Tree**: Preserved user modification in `chakrview/runtime/session.py` and untracked files across earlier workbenches.
- **Canonical Checkpoint Hash**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - **Calculated SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Bit-exact match).
- **Focused Regression Tests**: 29/29 tests in `tests/test_batch103_106_slot_weighted_sft.py` passed in 3.77s.

---

## 2. In-Depth Code & Implementation Audit

We audited `stream_experiment.py`, `slot_weighted_training.py`, and `eval_teacher_probs.py`:

### A. Anchor Token IDs (`37, 100, 101, 544`)
- **Tokens in Tokenizer**:
  - `37` $\to$ `'"'`
  - `100` $\to$ `'a'`
  - `101` $\to$ `'b'`
  - `544` $\to$ `'": '`
- **Validation**:
  - `tokenizer.encode('"a": ')` produces `[37, 100, 544]`.
  - `tokenizer.encode('"b": ')` produces `[37, 101, 544]`.
  - *Finding*: These IDs are safe, robust anchor sequences under the canonical BPE tokenizer (`vocab_4096`).

### B. Target Slot Discovery & Multi-Token Boundary Handling
- In `stream_experiment.py`, the scan rule `while j < len(tokens) and tokens[j] != TOK_QUOTE and j < i + 6:` was audited:
  - Total tokens in stream: 71,139.
  - Exactly 391 `a` fields and 391 `b` fields were located (matching the exact 392 conversations in the stream).
  - Spurious anchor tokens marked as slots: **0**.
  - No prompt tokens or adjacent record tokens were marked as slots.
  - **Audit Discovery on Boundary**: For `b` value tokens, the trailing terminator in the JSON string is `}` (`128`), not `"` (`37`). Because `tokens[j] != 37` held until `j < i + 6`, the loop marked the value token and the trailing `}` (`128`) and `<` (`63`).
  - *Impact*: In the stream experiment, the slot weight ($5\times$) weighted not only the numeric digit, but also the trailing JSON brace and turn delimiter `<|im_end|>`. This explains why the model in Batch 105 shifted its whitespace/delimiter tokens while maintaining the static attractor mode (`3, 8`).

### C. Shifted-Label Indexing and Loss Alignment
- In `stream_experiment.py`:
  - `x = tokens[i : i + seq_len]`, `y = tokens[i + 1 : i + seq_len + 1]`.
  - At position $t \in [0, \text{seq\_len}-1]$, model input $x[t]$ predicts target $y[t] = \text{tokens}[i + 1 + t]$.
  - The weight vector `w_mat[b, t]` inspected `slot_mask[i + 1 + t]`.
  - *Finding*: Causal shifted-label alignment was mathematically correct; the loss term predicting target $y_t$ received weight $w_t$.

### D. Controlled Comparison Validity
- Condition B (Uniform Control) and Condition C (Slot-Weighted 5×) started from the identical canonical checkpoint (`training_checkpoint_step_10000.pt`), using the identical stream, identical random seed (`seed=42`), identical optimizer hyperparameters, and identical token budget (25,600 token exposures).
- Condition A (historical Condition C baseline) was trained under a different run of the same objective; its metrics matched Condition B bit-for-bit (100% syntax, 0% exact pair on train/val, 5.6% accidental test collision).

### E. Prompt Target Isolation and Metric Denominators
- Evaluation prompts across train, val, test, and clarification splits contained strictly the user query and instructions, ending at `<|im_start|>assistant\n`. Target completions and answers were never injected into the inference context.
- Constant-call collapse was computed as $\max(\text{count}) / N$ across all evaluated samples, properly reflecting output variety.

---

## 3. Audit Conclusions & Summary

1. **Was the weighted-loss experiment mathematically sound?**
   Yes, the causal shifted-label alignment and controlled arm initialization were sound.
2. **Did slot-weighting fail due to implementation error or model behavior?**
   While the slot scan boundary included trailing `}` tokens, both the per-record script (`slot_weighted_training.py`) and stream script demonstrated that applying loss penalties to argument slots alone does not teach cross-attention transport across long, syntax-heavy templates.
3. **Valid Conclusions from Previous Sprint**:
   - Scalar loss weighting does not solve input-conditioned dynamic argument generation.
   - SFT on pure tool examples causes 100% spurious tool calling on out-of-domain clarification controls.
   - Accidental test accuracy ($5.56\%$) was an artifact of static attractor collision (`1, 8`).
