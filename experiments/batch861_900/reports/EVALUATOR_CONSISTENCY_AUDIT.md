# WP 861–865: Evaluator Consistency Audit

## 1. Scope and Objective
Audit the evaluation pipeline in `evaluate_combined_pointer_generalization.py` and `evaluate_argument_fidelity.py` to ensure consistency in:
- Prompt extraction and string normalization.
- Tokenizer encoding boundaries.
- Error categorization (syntax failure vs argument mismatch).
- Scoring neutrality without artificial gold patching.

---

## 2. Findings

### 2.1 Prompt Extraction & Boundaries
- Evaluators cleanly split input prompt at `\nPlan:\n` to isolate prompt context from the model's autoregressive outputs.
- Prompt token length $P$ is explicitly recorded to prevent pointers from referencing future or generated tokens.

### 2.2 Distinguishing Number Types
- Evaluators distinguish literals provided directly in user text from intermediate computed references (`$0`, `$1`).
- The parser strictly marks intermediate step outputs with the `$` prefix in `PlanDSLParser`.

### 2.3 Handling of Decimals, Negatives, and Unicode
- The DSL lexer correctly supports negative integers (`-14`), decimals, and multi-word quoted strings (`'Decentralized...'`).
- However, because token-level pointers operate on individual BPE fragments, strings with punctuation or multi-byte Unicode characters suffer catastrophic fragmentation during token-by-token emission.

### 2.4 Error Categorization Verification
- The evaluation cleanly bifurcates:
  1. **Syntax Failure**: The generated text failed regex grammar or terminal `<END_PLAN>` state.
  2. **DAG / Semantic Failure**: Circular dependency or invalid step reference.
  3. **Argument Copy Failure**: Syntax was valid, but extracted arguments do not match prompt ground-truth operands.

### 2.5 Conclusion
The evaluation pipeline is mathematically and semantically sound. It does not introduce artificial bias, nor does it falsely penalize valid variations. The 0/42 copy failure is an accurate reflection of the token-level pointer's inability to span multi-token BPE literals.
