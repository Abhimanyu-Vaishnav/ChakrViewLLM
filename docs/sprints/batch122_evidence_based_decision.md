# Batch 122 — Evidence-Based Decision on Tool-Gating and Argument Binding

## Verification of Mandatory Non-Negotiables & Repository State

- **Git Branch**: `dev/v0.6-v0.7-validation`
- **Git HEAD**: `b4e0189a9f1ccf4466137db11d9647ec8d473d04`
- **Working-Tree Integrity**: User changes in `chakrview/runtime/session.py` and existing Workbench files preserved without modification or automatic staging.
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - File size: 42,212,787 bytes
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-exact).
- **Execution Constraints**: Zero external pretrained models, Hugging Face wrappers, or hosted LLM APIs were used. Tool execution is strictly confined to the deterministic allowlisted calculator runtime (`validate_and_execute_tool_call`).

---

## Controlled Decision

Based on the empirical evidence gathered across Batches 119–121, we select:

### **Conclusion**: Useful but limited improvement, with residual errors.

#### Evidence Justification:
1. **Survives Test-Leakage Audit on Unseen Operands**:
   When evaluated on the 15 single-digit operand pairs that were demonstrably absent from all historical training splits (Batches 95, 108, and 116), the curriculum bridge model achieved **100.0% exact-call accuracy (15/15)** under the canonical prompt template. In contrast, the canonical base model under identical curriculum supervision scored only **13.3% (2/15)** due to constant-output collapse.
2. **Robust Tool-Gating / Zero Spurious Calls**:
   Both the canonical control and the curriculum bridge achieved a **0.0% false-positive rate (0/10)** across negative controls and underspecified additions, with **100.0% clean refusals**. Neither valid nor malformed tool calls were emitted when tool calls were inappropriate.
3. **Residual Prompt-Phrasing Fragility**:
   When presented with **fresh, unseen prompt phrasings** (e.g. `"Please add A={a} and B={b}."` or `"Find the total sum of A={a} and B={b}."`), exact-call accuracy dropped from 100.0% to **20.0% (2/10)**. While operand $B$ was extracted reliably (88% accuracy), operand $A$ defaulted to memorized single-digit constants (2 or 3) when the prompt syntax deviated from the multi-line `Calculate sum of:\nA=..\nB=..` format.
4. **Conclusion on Mechanism**:
   The copy-task curriculum pre-training successfully induces an operand-binding pathway in the transformer that transfers to tool-argument emission without memorizing specific number pairs. However, this binding circuit is partially conditioned on the positional and syntactic structure of the prompt template.

---

## Architectural Verdict & Next Benchmark Specification

### Architecture Decision:
- **Do NOT introduce a pointer network or custom copy attention layer yet.**
  The standard attention architecture of `ChakrMicro v0.1` has proven capable of learning abstract, generalized copy-and-bind operations across unseen numerical tokens without custom inductive biases.
- **Do NOT expand to multi-operation arithmetic prematurely.**
  The prompt-phrasing fragility must be addressed through syntactic diversity training before compounding complexity with multiple tools.

### Next Benchmark Specification: `batch123_prompt_diverse_multi_digit_benchmark`
To be evaluated in subsequent sprints as a separate, isolated benchmark:
1. **Syntactic Diversity**: Incorporate 8 distinct prompt templates into training rather than a single fixed multi-line format.
2. **Multi-Digit Arithmetic Evaluation**: Formulate an isolated test set of 2-digit numbers ($10..99$) with unseen prompt templates, preserving clean separation between single-digit training and multi-digit compositional evaluation.
