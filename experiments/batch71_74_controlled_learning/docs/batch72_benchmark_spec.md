# Batch 72 — Trustworthy Arithmetic Evaluation & Benchmark Specification

## 1. Benchmark Definition and Invariants

To evaluate whether ChakrMicro can learn an addition transformation, Batch 72 establishes a bounded, deterministic single-digit addition benchmark with zero multi-token tokenizer artifacts:

- **Mathematical Domain**: $a + b = c$, where $a, b \in \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$ such that $c = a + b \le 9$.
- **Total Possible Pairs**: Exactly 55 unique ordered pairs $(a, b)$.
- **Token Invariant**: Every input operand $a, b$ and output answer $c$ corresponds to exactly one token ID in ChakrMicro's vocabulary ($0 \to 51, \dots, 9 \to 60$). No digit splitting or multi-token BPE merging occurs.
- **Prompt Format**: Standard ChatML assistant prompt:
  ```
  <|im_start|>user\nAdd: a + b = ?<|im_end|>\n<|im_start|>assistant\n
  ```
- **Response Format**:
  ```
  c<|im_end|>\n
  ```

---

## 2. Dataset Splits and Disjoint Integrity

Using deterministic random assignment (Seed = 42), the 55 pairs are partitioned as:
1. **Training Set (`train_arithmetic.jsonl`)**: 30 pairs. SHA-256: `a0bd411bca4c67f964fb5a78d572b9e4597d83323cf3b09560a8b50cecd57b97`.
2. **Validation Set (`val_arithmetic.jsonl`)**: 10 pairs. SHA-256: `1ada6d162c91b082c0fd66b0a5bd96936c6d67e0367365e9faac62b933dda5c7`.
3. **Held-Out Test Set (`test_held_out.jsonl`)**: 10 pairs. SHA-256: `e4ed6f25eb385fae500e18b45cebf9f5a0bd1fa0102071bf8806d8b4893a8f07`.

### Split Integrity Verification:
- Overlap (Train $\cap$ Val): **0 pairs**.
- Overlap (Train $\cap$ Test): **0 pairs**.
- Overlap (Val $\cap$ Test): **0 pairs**.
- All splits are strictly disjoint at the operand-pair level. Held-out test records are strictly isolated from training, model selection, and hyperparameter tuning.

---

## 3. Symmetry Diagnostic Specification

In addition to standard held-out evaluation, Batch 72 introduces a **Symmetry / Commutativity Diagnostic** (`symmetry_diagnostic.jsonl`, 11 pairs, SHA-256: `ab2dd2b83ffe1b9eabaf7099ce05b2859eaf2bdfb879d4ded7f31350895c2453`):
- Pairs $(a, b)$ in Validation or Test whose reverse pair $(b, a)$ was present in the Training set (with $a \neq b$).
- **Purpose**: Evaluates whether the model learns the commutative property of addition ($a + b = b + a$). If a model memorizes $(3, 4) \to 7$ but cannot compute $(4, 3) \to 7$, this proves memorization of string tokens rather than an arithmetic operator.

---

## 4. Evaluation and Scoring Protocols

1. **Exact-Match Answer Accuracy**:
   - The generated response is stripped of `<|im_end|>` and leading/trailing whitespace.
   - Evaluated as `clean_generated == expected_answer`.
2. **EOS Completion Rate**:
   - Evaluated separately from answer correctness. A model that generates the correct digit but does not generate `<|im_end|>`/EOS is marked correct answer but incomplete EOS. A model that emits wrong digits followed by `<|im_end|>` is marked incorrect answer but complete EOS.
3. **Teacher-Forced Answer Token Loss**:
   - Evaluated specifically at the answer token position ($|P|$), separate from the 8 ChatML template tokens.
