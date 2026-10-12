# Batch 96: Operand Slot-Binding Experimental Results

## 1. Experimental Conditions
- **Condition B (Reference)**: Unstructured natural-language tool-calling model from Batches 91–94 (`condition_b_tool_calling_step_100.pt`, SHA-256: `65eb913bbed9c855f17eb07fabfb5ff6a30843d472f467ac37a11d6731505b41`).
- **Condition C (Slot-Binding SFT)**: Trained for 100 optimizer steps on 71,139 tokens of structured slot-binding format (`condition_c_slot_binding_step_100.pt`, SHA-256: `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`).

## 2. Quantitative Results Comparison

| Condition | Split | Valid Tool Syntax | Runtime Exec Rate | Correct Operand A | Correct Operand B | Exact Argument Pair | Constant-Call Collapse | End-to-End Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Step 10k)** | Train (64) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51.56% | 0.0% |
| | Val (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 55.56% | 0.0% |
| | Test (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 72.22% | 0.0% |
| **Condition B (NL Tool)** | Train (64) | **100.0%** | **100.0%** | 3.12% | 20.31% | **1.56%** | **100.0%** (`{"a":1,"b":1}`) | **9.38%** |
| | Val (18) | **100.0%** | **100.0%** | 0.0% | 11.11% | **0.0%** | **100.0%** (`{"a":1,"b":1}`) | **11.11%** |
| | Test (18) | **100.0%** | **100.0%** | 0.0% | 0.0% | **0.0%** | **100.0%** (`{"a":1,"b":1}`) | **0.0%** |
| **Condition C (Slot-Binding)**| Train (64) | **100.0%** | **100.0%** | 1.56% | 7.81% | **0.0%** | **100.0%** (`{"a":1,"b":8}`) | **10.94%** |
| | Val (18) | **100.0%** | **100.0%** | 0.0% | 0.0% | **0.0%** | **100.0%** (`{"a":1,"b":8}`) | **0.0%** |
| | Test (18) | **100.0%** | **100.0%** | 5.56% | 16.67% | **5.56% (1/18)** | **100.0%** (`{"a":1,"b":8}`) | **0.0%** |

## 3. Scientific Findings & Mode Collapse Diagnosis
1. **Shifting Mode Attractor**:
   - Condition B collapsed entirely to `TOOL_CALL: {"tool": "calculator", "op": "add", "a": 1, "b": 1}`.
   - Condition C collapsed entirely to `TOOL_CALL: {"tool": "calculator", "op": "add", "a": 1, "b": 8}`.
2. **Held-Out Test Accuracy**:
   - Condition C scored 1/18 (5.56%) on exact argument pair accuracy on the held-out test set solely because the pair `(1, 8)` was present in the test set.
   - The collapse rate remained **100.0%** across train, validation, and test sets.
3. **Conclusion**:
   - Providing explicit operand slots (`Operand A: {a}`, `Operand B: {b}`) did **not** resolve the mode-collapse.
   - The model at 3.4M parameters treats the tool-call generation as emitting an invariant procedural macro token sequence rather than executing cross-attention slot-filling from the prompt context.
   - **Classification**: **No demonstrated dynamic generalization capability** beyond syntax memorization.
