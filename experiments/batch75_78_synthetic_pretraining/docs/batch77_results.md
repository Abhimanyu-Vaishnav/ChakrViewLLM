# Batch 77 — Controlled Pretraining and Curriculum Comparison Results

## 1. Experimental Overview & Setup

Batch 77 conducted a resource-bounded pretraining and curriculum comparison to evaluate whether structured mathematical pretraining improves held-out arithmetic reasoning in ChakrMicro v0.1:
- **Starting Weights**: Canonical Checkpoint Step 10k (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Condition A (Language Control)**: 100 steps on 56,231 tokens from Stage C pretraining corpus.
- **Condition B (Arithmetic Curriculum)**: 100 steps on 56,231 tokens from multi-template synthetic addition data (exclusively using the 30 allowed training pairs).
- **Overfit Control**: 50 steps on 3 training examples (`3 + 3 = 6`, `2 + 4 = 6`, `7 + 0 = 7`) to verify backpropagation and optimization health.
- **Hardware**: CPU-only execution (Intel Core i5-1135G7, 31.64 GB RAM, PyTorch 2.14.0+cpu).

---

## 2. Quantitative Results Across All Evaluated Conditions

| Condition / Strategy | TRAIN Exact Match | VAL Exact Match | TEST Exact Match | Symmetry Diagnostic | TRAIN Ans Loss | VAL Ans Loss | TEST Ans Loss | EOS Rate | Modal Prediction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 10.0813 | 10.0336 | 10.2717 | 0.0% | Wikitable repetitive loops |
| **Condition A (Control)** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 11.2043 | 11.0931 | 11.1422 | 0.0% | Wikitable headers |
| **Condition B (Curriculum)** | 3 / 30 (10.0%) | 2 / 10 (20.0%)* | 2 / 10 (20.0%)* | 2 / 11 (18.2%)* | **2.3183** | **2.0942** | **2.1251** | **100.0%** | **`6` (100% of outputs)** |
| **Overfit Control (3 exs)** | 5 / 30 (16.7%) | 0 / 10 (0.0%) | 3 / 10 (30.0%)* | 1 / 11 (9.1%)* | 9.2810 | 10.0986 | 7.7603 | 100.0% | `6` (77%) and `7` (23%) |

*\*Note on Generalization*: The apparent 2/10 validation and 2/10 held-out test exact-match accuracy in Condition B is **100% due to modal collapse onto the digit `6`**.

---

## 3. Detailed Diagnosis: The `6` Modal Attractor

Examination of every single generation across all splits in Condition B reveals:
- **Train (30 pairs)**: Model generated `6` for all 30 prompts $\to$ Exactly 3 train pairs have expected sum 6 (`3+3`, `2+4`, `5+1`), giving 3/30 exact match.
- **Validation (10 pairs)**: Model generated `6` for all 10 prompts $\to$ Exactly 2 validation pairs have expected sum 6 (`4+2`, `6+0`), giving 2/10 exact match.
- **Held-Out Test (10 pairs)**: Model generated `6` for all 10 prompts $\to$ Exactly 2 test pairs have expected sum 6 (`0+6`, `1+5`), giving 2/10 exact match.
- **Symmetry Diagnostic (11 pairs)**: Model generated `6` for all 11 prompts $\to$ Exactly 2 pairs have expected sum 6 (`4+2`, `1+5`), giving 2/11 exact match.

### Scientific Finding:
Continual synthetic pretraining successfully achieved:
1. **ChatML Template Mastery**: Template token cross-entropy dropped from 6.36 to **0.0016**; EOS completion reached **100.0%**.
2. **Format Adherence**: The model transitioned from generating unstructured HTML/wikitable boilerplate (`|- \n|200...`) to producing clean, single-token numerical outputs followed immediately by `<|im_end|>`.
3. **Distributional Learning Without Functional Reasoning**: Cross-entropy at the answer position dropped from ~10.1 to **~2.12**. However, instead of computing $f(a, b) = a + b$, the model converged to the mean/mode of the training sum distribution (predicting `6` uniformly). It did not generalize the addition operator to held-out pairs.

---

## 4. Checkpoint Invariants and Cold-Reload Verification

All checkpoints were persisted with complete metadata and verified with fresh-process cold reload:
- Baseline Step 10k SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Condition A Checkpoint SHA-256: `65eb99c293eb6df5d517b0e2e0da6fbff122c07e45b3a4f918bf831571bc4514`
- Condition B Checkpoint SHA-256: `04c70fe7ef5b1a5bec8085b97a7d6c8f72c1fb6689b0a6a8ca4eddeffe30b2e4`
- Overfit Control Checkpoint SHA-256: `1e95fa0a944dc618db05b3e100dc065e1289196b0559f977b3b5550a1df2809e`

Cold reload of Condition B reproduced exact test match count (2/10) and mean answer-token loss (2.1251) bit-for-bit.
