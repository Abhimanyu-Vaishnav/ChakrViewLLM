# Batch 92: Bounded Tool-Calling Learning Experiment Results

## 1. Experimental Conditions & Setup
- **Base Model**: Canonical `ChakrMicro v0.1` step-10,000 checkpoint (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Condition A (Direct Arithmetic SFT)**: Trained for 100 optimizer steps on 71,139 tokens of direct addition format (`answer: <sum>`). Checkpoint SHA-256: `97c21db79ac763720b2a62e130c3df1f4808ee3a46c14bc365dfdbeaeb21633d`.
- **Condition B (Tool-Calling SFT)**: Trained for 100 optimizer steps on 71,139 tokens of multi-turn tool-calling format (`TOOL_CALL: {"tool": "calculator", "op": "add", "a": X, "b": Y}`). Checkpoint SHA-256: `65eb913bbed9c855f17eb07fabfb5ff6a30843d472f467ac37a11d6731505b41`.
- **Dataset Partitioning**: Balanced 0–9 operand benchmark with reversed-pair grouping (zero pair leakage):
  - Train: 64 examples (32 operand pairs)
  - Validation: 18 examples (9 held-out operand pairs)
  - Held-Out Test: 18 examples (9 held-out operand pairs)
  - Clarification Controls: 5 examples (unsupported general language & ambiguous requests)

## 2. Quantitative Results

| Metric | Condition | Train (n=64) | Val (n=18) | Test (n=18) | Clarification Control (n=5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Valid Tool Call Syntax** | Baseline Step 10k | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | 100.0% safe non-tool |
| | Condition A (Direct) | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | 100.0% safe non-tool |
| | **Condition B (Tool Call)** | **100.0% (64/64)** | **100.0% (18/18)** | **100.0% (18/18)** | 0.0% safe (spurious call) |
| **Correct Tool Selection** | Baseline Step 10k | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | Condition A (Direct) | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition B (Tool Call)** | **100.0% (64/64)** | **100.0% (18/18)** | **100.0% (18/18)** | N/A |
| **Runtime Execution Rate** | Baseline Step 10k | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | Condition A (Direct) | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition B (Tool Call)** | **100.0% (64/64)** | **100.0% (18/18)** | **100.0% (18/18)** | N/A |
| **Correct Operands Generated** | Baseline Step 10k | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | Condition A (Direct) | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition B (Tool Call)** | **1.56% (1/64)** | **0.0% (0/18)** | **0.0% (0/18)** | N/A |
| **End-to-End Success** | Baseline Step 10k | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | Condition A (Direct) | 0.0% (0/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition B (Tool Call)** | **9.38% (6/64)** | **11.11% (2/18)** | **0.0% (0/18)** | N/A |

## 3. Detailed Diagnosis & Canned-Call Collapse Analysis
1. **Tool Schema Acquisition**: The model successfully learned the exact structural boilerplate syntax:
   `TOOL_CALL: {"tool": "calculator", "op": "add", "a": X, "b": Y}`
   Achieving **100% valid JSON schema validation** and **100% runtime execution** across train, validation, and test sets.
2. **Canned Argument Collapse**: On the test set, the model generated `TOOL_CALL: {"tool": "calculator", "op": "add", "a": 1, "b": 1}` for 100% (18/18) of prompts. On validation, it alternated between `[1, 1]` and `[1, 9]`.
3. **Generalization Classification**: **Partial capability**. While schema compliance and deterministic tool execution were fully acquired, operand conditioning collapsed to mode-seeking template values rather than dynamic argument binding from user prompts.
