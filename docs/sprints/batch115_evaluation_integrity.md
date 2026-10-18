# Batch 115 — Evaluation Semantics and Metric Integrity Audit

**Sprint**: ChakrView Master Sprint Batches 115–118  
**Module**: Tool-Gating Calibration, Strict Execution Metrics, and Reliable Argument Binding  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Audit of Baseline State & Previous Evaluation Weaknesses

Before modifying code, we verified:
- **Branch**: `dev/v0.6-v0.7-validation`
- **Current HEAD**: `b4e0189a9f1ccf4466137db11d9647ec8d473d04`
- **Canonical Step-10k Checkpoint**:
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Size: 42,212,787 bytes
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Bit-exact match verified).

### Identified Weaknesses in Prior Evaluation Scripts:
1. **Accidental Arithmetic Match Risk**:
   - In earlier evaluation loops, an execution was counted as successful if `result == expected_sum`.
   - *Flaw*: For query $1 + 8 = 9$, if the model hallucinated $5 + 4$, the result was $9$, which would have erroneously scored as end-to-end correct despite incorrect arguments.
2. **Conflation of Syntax Validity and Schema Conformity**:
   - Parsing valid JSON does not guarantee that required fields (`tool`, `op`, `a`, `b`) exist, nor does it check whether prohibited injection fields are present.
3. **Absence of Negative Controls in Training**:
   - Prior sprints evaluated on 5 clarification controls without ever exposing the model to negative examples during SFT, guaranteeing 100% false-positive tool selection.

---

## 2. Rigorous 11-Dimension Metric Specification

In `experiments/batch115_118_tool_gating_curriculum/metrics.py`, we implemented:

1. **Parse Rate**: Candidate string strictly matches `TOOL_CALL: {json}` grammar.
2. **Schema Validity**: Parsed dictionary contains strictly `{"tool", "op", "a", "b"}`, prohibits extra fields, and enforces integer types on operands.
3. **Tool Accuracy**: `tool == "calculator"`.
4. **Operation Accuracy**: `op == "add"`.
5. **Operand A Accuracy**: `a == expected_a`.
6. **Operand B Accuracy**: `b == expected_b`.
7. **Exact Call Accuracy**: Tool, operation, A, and B all simultaneously match.
8. **Execution Success**: The allowlisted runtime accepts and safely executes the call.
9. **End-to-End Correctness**: Exact call accuracy is true, runtime executes successfully, and arithmetic result strictly matches expected ground truth.
10. **False-Positive Tool-Call Rate**: Non-tool queries (clarifications, out-of-domain requests, underspecified arithmetic) triggering an attempted tool call.
11. **Collapse Rate**: Frequency of the most common normalized output candidate.

---

## 3. Unit Test Verification

Deterministic unit tests in `tests/test_batch115_evaluation_semantics.py`:
- Verified that accidental sum collisions (`5 + 4 = 9` for `1 + 8`) are rejected with `e2e_correct=False`.
- Verified strict rejection of extra keys, type errors, and malformed grammar.
- Verified false-positive tool call detection on non-tool records.
- All 6 unit tests pass in 0.06s without relying on neural checkpoint weights.
