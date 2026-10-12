# Batch 91: Explicit Calculator Tool Protocol Specification

## 1. Overview & Objective
Batch 91 formalizes an explicit, safe, and schema-governed tool-calling protocol for ChakrView (`ChakrMicro v0.1`). The design goal is to evaluate whether the neural model can learn to emit structured tool requests, handle deterministic software outputs, and formulate answers, without using external model APIs, runtime regex repairs, or arbitrary code execution.

## 2. Tool-Call Protocol Schema

### Assistant Invocation Turn (Turn 1)
When presented with a user calculation request, the model emits a structured tool request:
```
<|im_start|>assistant
TOOL_CALL: {"tool": "calculator", "op": "add", "a": 3, "b": 4}<|im_end|>
```

### JSON Schema Specification
- `tool` (string, required): Allowed tools = `["calculator"]`.
- `op` (string, required): Allowed operations = `["add", "sub", "mul"]`.
- `a` (integer, required): Allowed bounds = `[-10000, 10000]`.
- `b` (integer, required): Allowed bounds = `[-10000, 10000]`.

### Deterministic Tool Response Turn
```
<|im_start|>tool
{"result": 7, "status": "SUCCESS"}<|im_end|>
```

### Assistant Final Response Turn (Turn 2)
```
<|im_start|>assistant
The answer is 7.<|im_end|>
```

## 3. Runtime Security Boundaries & Defense In Depth
1. **Zero Execution (`eval`/`exec`)**: Python's `eval()` and `exec()` are strictly prohibited. The calculator module executes pure integer operations with type checks (`isinstance(a, int) and not isinstance(a, bool)`).
2. **Schema & Allowlist Enforcement**: Requests specifying unknown tools (e.g., `bash`, `python`) or operations (e.g., `div`, `exp`) are rejected with explicit failure codes:
   - `MALFORMED_JSON`: Invalid syntax.
   - `MISSING_FIELD`: Missing mandatory schema fields.
   - `UNSUPPORTED_TOOL`: Tool not in registered allowlist.
   - `UNSUPPORTED_OPERATION`: Operation not in registered allowlist.
   - `OPERAND_TYPE_ERROR`: Non-integer operand passed.
   - `OPERAND_BOUNDS_ERROR`: Operand exceeded $[-10000, 10000]$.
3. **Turn & Cycle Caps**: The conversation loop strictly caps tool cycles to 1 turn to prevent infinite recursions.
4. **Zero Silent Repair**: The runtime does not inspect user prompts to patch incorrect model arguments or syntax. Malformed or incorrect calls are recorded faithfully as failures.
5. **Provenance Tagging**: All executed calculator results are tagged with `"provenance": "DETERMINISTIC_SAFE_CALCULATOR"` to maintain clear separation from model intelligence.
