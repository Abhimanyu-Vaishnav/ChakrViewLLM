# Batch 76 — Synthetic Data and Experiment Design Specification

## 1. Experimental Objective and Hypotheses

The objective of Sprint Batches 75–78 is to evaluate whether structured synthetic mathematical pretraining can improve ChakrMicro v0.1's capability to solve unseen single-digit addition problems, or whether the model remains constrained to memorization and modal collapse.

### Scientific Hypotheses:
1. **$H_0$ (Modal Collapse / Memorization Null)**: Continual pretraining on mathematical curriculum data will reduce answer-token cross-entropy loss on trained examples, but greedy generation on held-out test operand pairs will remain at or near zero exact match (or collapse to a single modal digit such as `7`).
2. **$H_1$ (Operator Generalization Alternative)**: Exposure to multi-template mathematical expressions, arithmetic drills, and causal equation structures enables the attention layers to compute the addition operator, resulting in statistically significant non-constant exact-match accuracy on held-out operand pairs without leaking training examples.

---

## 2. Strict Split Design & Zero-Contamination Guarantees

The evaluation benchmark established in Batch 72 consists of single-digit addition pairs where $a + b \le 9$ ($a, b \in \{0, \dots, 9\}$):
- **Train Set**: Exactly 30 unique ordered pairs $(a, b)$.
- **Validation Set**: Exactly 10 unique ordered pairs $(a, b)$.
- **Held-Out Test Set**: Exactly 10 unique ordered pairs $(a, b)$.
- **Symmetry Diagnostic Set**: Exactly 11 pairs $(a, b)$ testing commutativity.

### Disjoint Isolation & Non-Contamination Invariant:
- To eliminate any test-set leakage, the Condition B synthetic pretraining generator draws exclusively from the **30 allowed training operand pairs**:
  `[(0, 0), (0, 3), (0, 4), (0, 9), (1, 0), (1, 1), (1, 6), (1, 8), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 1), (3, 2), (3, 3), (3, 6), (4, 1), (4, 4), (5, 1), (5, 2), (5, 4), (6, 1), (7, 0), (7, 1), (7, 2), (8, 1), (9, 0)]`
- The 10 validation pairs and 10 held-out test pairs are strictly isolated:
  * $\text{Pretraining Pairs} \cap \text{Val Pairs} = \emptyset$
  * $\text{Pretraining Pairs} \cap \text{Test Pairs} = \emptyset$

---

## 3. Multi-Template Synthetic Curriculum (Condition B)

To avoid superficial prompt overfitting and encourage causal arithmetic structure learning, the synthetic data generator (`generate_curriculum_data.py`) renders training equations into 8 diverse syntactic structures:
1. **ChatML Assistant Format**: `<|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>\n`
2. **Equation**: `Equation: {a} + {b} = {c}.\n`
3. **Plain Expression**: `{a} + {b} = {c}\n`
4. **Word Problem**: `If you have {a} apples and receive {b} more, you have {c} apples in total.\n`
5. **Question and Answer**: `Question: What is {a} plus {b}? Answer: {c}.\n`
6. **Sum Query**: `The sum of {a} and {b} is equal to {c}.\n`
7. **Arithmetic Drill**: `Arithmetic Drill: {a} + {b} equals {c}.\n`
8. **Calculation Identity**: `Calculate {a} + {b}. Result: {c}.\n`

Each document is tokenized using the frozen canonical 4096-vocab BPE tokenizer with end-of-sequence delimiter (`EOS = 1`).
- Total generated documents: 2,500.
- Total token count: **56,231 tokens**.
- SHA-256: `d277863c169079d443b3232134fd69a810dbdde0350b7de721aa9891a8d28075`.

---

## 4. Matched Generic Language Control (Condition A)

To establish an unconfounded experimental control, Condition A uses the exact same token budget extracted from the existing canonical Stage C pretraining corpus (`data/tokenized/stage_c/train/shard_00000.bin`):
- Total extracted tokens: exactly **56,231 tokens** (1:1 token-count match with Condition B).
- SHA-256: `2076c9c543b41d526d858dfc9013d50e0d93a2465e70198e43b13a46294024fa`.

Both conditions start from the identical verified checkpoint:
`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).

---

## 5. Controlled Experimental Parameters

| Hyperparameter | Condition A (Language Control) | Condition B (Arithmetic Curriculum) |
|:---|:---|:---|
| **Starting Checkpoint** | Step 10,000 Canonical Checkpoint | Step 10,000 Canonical Checkpoint |
| **Model Architecture** | ChakrMicro v0.1 (3,443,136 parameters) | ChakrMicro v0.1 (3,443,136 parameters) |
| **Token Corpus Size** | 56,231 tokens | 56,231 tokens |
| **Batch Size** | 4 | 4 |
| **Sequence Length** | 64 | 64 |
| **Training Steps** | 100 steps | 100 steps |
| **Total Tokens Trained** | 25,600 tokens ($100 \times 4 \times 64$) | 25,600 tokens ($100 \times 4 \times 64$) |
| **Optimizer** | AdamW (`lr=5e-4`, `weight_decay=0.01`) | AdamW (`lr=5e-4`, `weight_decay=0.01`) |
| **Gradient Clipping** | 1.0 | 1.0 |
| **Random Seed** | 42 | 42 |
| **Device** | CPU (Intel Core i5-1135G7) | CPU (Intel Core i5-1135G7) |

---

## 6. Pre-Registered Decision Criteria

1. **Non-Trivial Accuracy**:
   - Held-out test exact-match accuracy must exceed $0/10$ with **non-constant answers**.
   - If the model predicts the same digit across all 10 held-out prompts, it is diagnosed as modal collapse, regardless of whether that digit hits a matching sum.
2. **Answer-Token vs Template-Token Loss**:
   - Model must show decreasing cross-entropy specifically at the answer token position ($|P|$).
3. **Reproducibility**:
   - Evaluated models must reproduce identical predictions upon cold process reload from saved checkpoints.
