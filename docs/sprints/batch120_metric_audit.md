# Batch 120 — Metric Semantics & Dataset Claims Audit

## Executive Summary

Batch 120 audited and corrected the evaluation metrics module in `experiments/batch115_118_tool_gating_curriculum/metrics.py` and its associated unit test suite in `tests/test_batch115_evaluation_semantics.py`.

The previous implementation suffered from several potential measurement flaws:
1. **Uncaught Malformed Tool Attempts**: If a model attempted a tool call on a negative control prompt (e.g. emitting `TOOL_CALL: {` or `{"tool": "calculator"`), but failed JSON grammar parsing, it would not be identified as a tool attempt and could erroneously be counted as a clean non-tool response.
2. **Conflated False-Positive Dimensions**: Valid tool calls and malformed tool calls emitted on non-tool prompts were not separated into distinct rates.
3. **Accidental Equivalence Checks**: Ensuring that matching a scalar output (e.g. producing $a=5, b=4$ for target $a=1, b=8$ where both sum to 9) cannot pass end-to-end correctness.
4. **Dataset Class Distribution Description**: The curriculum training dataset in Batch 116 consists of 80% tool-invocation examples and 20% negative/non-tool control examples. Any previous characterization of the dataset as 50/50 balanced applied only to historical high-level curriculum designs, whereas the actual JSONL distribution is 80/20.

---

## Metric Corrections Implemented

### 1. Distinct Tool Attempt Detection
We implemented `detect_tool_call_attempt(text: str) -> bool` using regex pattern scanning:
```python
TOOL_ATTEMPT_PATTERNS = [
    re.compile(r"TOOL_CALL\s*:", re.IGNORECASE),
    re.compile(r'\{\s*"tool"\s*:', re.IGNORECASE),
    re.compile(r'\{\s*"op"\s*:', re.IGNORECASE),
    re.compile(r"<tool_call>", re.IGNORECASE),
]
```
Even if JSON parsing fails (`is_parsed = False`), if `detect_tool_call_attempt` returns `True` on a non-tool prompt (`requires_tool = False`), the prediction is immediately flagged:
- `attempted_tool = True`
- `false_positive = True`
- `malformed_false_positive = True`
- `correct = False`

### 2. Eleven Decoupled Evaluation Dimensions
The metrics module now measures eleven separate dimensions with explicit denominators:
1. **Denominator**: Total number of examples in split ($N$).
2. **Parse Rate**: Percentage of completions conforming to `TOOL_CALL: {JSON}` format.
3. **Schema Validity Rate**: Percentage of parsed payloads satisfying exact allowlisted keys (`{"tool", "op", "a", "b"}`), correct integer types, and no prohibited extra keys.
4. **Tool Accuracy**: Ground truth matching `"calculator"`.
5. **Operation Accuracy**: Ground truth matching `"add"`.
6. **Operand A Accuracy**: Integer $a$ strictly matching expected input $a$.
7. **Operand B Accuracy**: Integer $b$ strictly matching expected input $b$.
8. **Exact-Call Accuracy**: Simultaneous match of tool, operation, $a$, and $b$.
9. **Execution Success Rate**: Tool call parsed and successfully executed in allowlisted runtime without exceptions.
10. **End-to-End Correctness**: Exact-call match AND execution success AND runtime result strictly equal to expected arithmetic outcome.
11. **Negative Control Robustness**:
    - `total_false_positive_rate`
    - `valid_false_positive_rate`
    - `malformed_false_positive_rate`
    - `clean_refusal_rate`

### 3. Swapped Operands & Accidental Matches
Unit tests explicitly guarantee:
- Swapped operands (e.g., predicted `{"a": 7, "b": 2}` for prompt requiring `[2, 7]`) fail `a_correct`, `b_correct`, `exact_call`, and `e2e_correct`.
- Accidental arithmetic result matches (e.g., predicting `{"a": 5, "b": 4}` summing to 9 when prompt required `[1, 8]`) fail `exact_call` and `e2e_correct`.

---

## Dataset Claim Verification

- **Curriculum Training Split**: 80 records total (64 tool additions, 16 non-tool prompts). Exactly **80.0% tool / 20.0% control**.
- **Historical Claims**: Sprints 111–114 labeled this a "balanced curriculum bridge", referring to balancing operand copy representations with tool templates. In terms of tool vs non-tool binary classification, it is an 80/20 imbalanced curriculum.

All regression tests pass.
