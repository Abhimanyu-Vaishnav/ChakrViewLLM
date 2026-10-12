# Batch 82 — Independent Audit, Engineering Decisions, and Release Readiness

## 1. Independent Audit of Experimental Validity

A systematic audit of Sprint Batches 79–82 was performed:

### 1.1 Unordered-Pair Disjointness
- All 100 single-digit addition pairs were partitioned into 55 unordered pairs $(u = (\min(a, b), \max(a, b)))$.
- 35 unordered pairs were allocated to Train (64 ordered pairs).
- 10 unordered pairs were allocated to Validation (18 ordered pairs).
- 10 unordered pairs were allocated to Held-Out Test (18 ordered pairs).
- $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Zero reversed-pair leakage confirmed**: Because $(a, b)$ and $(b, a)$ always belong to the same split, no reversed pair of a held-out test item was present in the training set.

### 1.2 Constant-Answer Baseline Evaluation
- In the training set, the most common answer was `6` (10.94% frequency).
- In both Validation and Held-Out Test, the constant-`6` baseline accuracy is **0.00%**.
- Condition C achieved 1/18 (5.56%) on test by predicting `10` on `2 + 8 = 10`. However, as proven in the first-operand mechanistic analysis, this was not general addition; it was a piecewise threshold heuristic mapping $a \in \{1, 2\} \to 10$.

### 1.3 Matched Token Budgets and Weight Initialization
- Baseline Step 10k SHA-256 verified: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
- Both Condition B (generic language control) and Condition C (balanced curriculum) started from this identical checkpoint.
- Both trained on exactly 63,794 tokens for 100 optimizer steps with batch size 4 and sequence length 64.

---

## 2. Scientific Synthesis Across Sprints 71–82

| Sprint | Dataset Setup | Observed Behavior | Root Cause |
|:---|:---|:---|:---|
| **Batches 71–74** (SFT Pilot) | Single-digit addition SFT (30 pairs, sums $\le 9$) | Predicted constant `7` everywhere | Model minimizes cross-entropy by converging to distribution mode. |
| **Batches 75–78** (Synthetic Pretraining) | Multi-template pretraining (sums $\le 9$) | Predicted constant `6` everywhere | Template mastery (EOS 100%, loss 0.0016), but unconditioned answer attractor. |
| **Batches 79–82** (Balanced Pretraining) | Uniform sum-class pretraining (sums $0 \dots 18$) | Broken into piecewise bins (`10`, `18`, `15`) based on first operand $a$ | Balancing broke the single attractor into piecewise thresholding on the first token, but failed to compose $(a, b)$. |

### Fundamental Theoretical Takeaway:
ChakrMicro (3.44M params, 6 layers, $d_{model}=192$) using standard direct-answer autoregressive generation (`Add: a + b = ?\nassistant\nc`) cannot compose two input operands in a single immediate generation step. The attention mechanism attends strongly to the first operand $a$ and ignores $b$, defaulting to coarse thresholding.

To achieve genuine mathematical generalization without scaling model size to hundreds of millions of parameters, the architecture requires an **autoregressive scratchpad / chain-of-thought sequence** (e.g. `Add 2 + 8. Count: 2, 3, 4, 5, 6, 7, 8, 9, 10. Sum: 10`), giving transformer layers the sequential steps necessary to route information from both operands.

---

## 3. Engineering Decisions & Next Steps

1. **Keep Baseline Immutable**: Canonical step-10,000 checkpoint and architecture remain untainted.
2. **Preserve User Code**: `chakrview/runtime/session.py` remains unstaged and uncommitted.
3. **Commit Sprint Artifacts**: Add tests and docs under `experiments/batch79_82_balanced_arithmetic/` and `tests/test_batch79_82_balanced_arithmetic.py`.
