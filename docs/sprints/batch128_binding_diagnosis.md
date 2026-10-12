# Batch 128 — Multi-Digit Operand Binding Failure Mechanism Diagnosis

## Executive Summary

Batch 128 conducted a rigorous diagnosis to determine why `ChakrMicro v0.1` can reliably generate syntactically valid calculator tool calls but struggles to conditionally bind unseen multi-digit input operands.

---

## 1. Hypotheses Tested

### Hypothesis 1: Token Alignment & BPE Subword Fragmentation
- **Finding**:
  - The ChakrView BPE tokenizer encodes 42 two-digit numbers as single tokens and 48 numbers as two subword tokens.
  - In target JSON serialization (`"a": 17, `), 63 out of 90 numbers (70%) merge with the trailing comma (e.g. `17, ` is token ID 3658).
  - *Empirical Test*: On the held-out test set, operand A accuracy was 36.0% (9/25) for comma-merged numbers and 30.0% (3/10) for unmerged numbers. Exact-call accuracy was 24.0% (6/25) for merged vs 10.0% (1/10) for unmerged numbers.
  - *Conclusion*: While subword fragmentation creates token variance, it is **not the primary bottleneck** preventing generalization.

### Hypothesis 2: Training Template Distribution & Syntactic Overfitting
- **Finding**:
  - Training used 8 prompt templates. However, each unique training pair was seen only 1.17 times on average across all 8 templates (50 pairs seen once, 10 pairs seen twice).
  - On held-out template 1 (`"Execute calculator addition: first={a}, second={b}."`), the copy-initialized model achieved **100% exact-call accuracy (5/5)** on familiar pairs.
  - On held-out template 2 (`"Determine arithmetic total for inputs: A is {a} and B is {b}."`), both models produced **100% clean refusals (0/10 tool calls)**. Because the phrase `"Determine arithmetic total"` lacks any key words present in the training set (`Calculate`, `sum`, `add`, `Compute`, `Find`, `operands`, `operation`), the model's gating head classified it as out-of-domain and declined to trigger a tool call.

### Hypothesis 3: Insufficient Pair Exposure vs Attractor Learning
- **Finding**:
  - In the single-digit benchmark (100 total pairs), training on 64 pairs covered 64% of the entire pair space.
  - In the two-digit benchmark ($90 \times 90 = 8,100$ total pairs), training on 60 pairs covered only **0.74%** of the pair space.
  - Without sufficient density, the model learns partial digit associations or defaults to frequent training operands rather than zero-shot compositional binding.

---

## 2. Diagnostic Summary

| Feature | Directly Demonstrated | Plausible / Inferred | Unknown / Rejected |
| :--- | :--- | :--- | :--- |
| **BPE Comma Merging** | 63/90 numbers merge with trailing comma in JSON target | Adds minor token-selection friction | Rejected as the primary cause of binding failure |
| **Operand Order Preservation** | 0 swapped arguments across all 35 test evaluations | Attention layers preserve relative positional order | Swapping hypothesis rejected |
| **Negative Control Robustness** | 100% clean refusal (0% false positives) | Gating head learns conservative decision boundary | No risk of indiscriminate tool calling |
| **Template Gating Fragility** | 100% refusal on held-out template 2 | Missing keyword triggers refusal | Architectural copy failure rejected for template 2 |
| **Subword Attention Attractors** | High frequency of training digits appearing in predictions | Partial attention heads copy tens or units digit correctly | Attractor hypothesis supported |
