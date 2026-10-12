# Batch 95: Operand Slot-Binding Protocol & Dataset Audit

## 1. Experimental Motivation & Hypothesis
In Batches 91–94, `ChakrMicro v0.1` achieved 100% syntactic schema compliance for tool calling, but collapsed to constant static calls `{"a": 1, "b": 1}`. 

**Hypothesis**: Does separating operand identification from tool-call syntax formulation by providing explicit structured input slots (`Operand A: {a}`, `Operand B: {b}`) enable the model to extract and dynamically bind argument values to the tool call schema?

## 2. Dataset Partitioning & Anti-Leakage Contracts
To prevent split leakage, the dataset preserves the exact 100-pair domain from Batches 79–94:
- **Train Set**: 64 examples (32 unordered operand pairs).
- **Validation Set**: 18 examples (9 held-out operand pairs).
- **Test Set**: 18 examples (9 held-out operand pairs).
- **Clarification Controls**: 5 negative/unsupported examples.
- **Reversed-Pair Invariant**: $(a, b)$ and $(b, a)$ strictly share the same partition.
- **Prompt Representation**:
  ```
  <|im_start|>user
  Request: add numbers
  Operand A: {a}
  Operand B: {b}
  Required action: calculator addition<|im_end|>
  <|im_start|>assistant
  ```
- **Zero Leakage**: The user prompt never contains the answer, the equation string, or the raw JSON tool call.
