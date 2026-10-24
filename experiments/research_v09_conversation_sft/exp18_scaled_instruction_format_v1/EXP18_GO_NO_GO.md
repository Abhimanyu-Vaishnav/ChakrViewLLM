# Exp18 Authorization Decision — Evidence-Gated Verdict

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/EXP18_GO_NO_GO.md`  
**Audit Reference**: Task Evaluator v0.1 Metric Reconciliation & Exp17 Verification  
**Date**: 2026-10-23  

---

## 1. Final Verdict

**FINAL STATUS**: `READY_FOR_EXP18_PREFLIGHT`

*(Notice: Training was NOT started in this mission. The experiment specification and preflight checklist are prepared and authorized strictly for a separate, subsequent preflight execution.)*

---

## 2. Gate Verification Summary

Under the mission governance rules, authorization requires passing three mandatory gates:

| Verification Gate | Required Condition | Actual Observed Result | Gate Status |
| :--- | :--- | :--- | :---: |
| **Gate 1: Metric Reconciliation** | 100% discrepancy-free reconciliation of all 25 Task Evaluator v0.1 metrics against raw JSON outputs | Independently reproduced across baseline and SFT outputs (35/40 scorable, 0.0% accuracy, 45.0% EOS, 13.3% Chat Gate trigram repetition, exactly 0 discrepancies) | **PASS** |
| **Gate 2: Test Suite Verification** | 100% test pass on all unit, regression, and evaluator suites with exit code 0 | 147/147 tests pass cleanly (16 Task Evaluator, 122 core regression, 9 Exp17 dedicated); 0 failures, 0 errors | **PASS** |
| **Gate 3: Protected Asset Integrity** | All 8 canonical model, tokenizer, runtime, and dataset SHA-256 hashes match expected bit-for-bit | All 8 hashes match expected values exactly; working tree clean of accidental modifications | **PASS** |

---

## 3. Scope of Authorization & Mandatory Guardrails

1. **Strictly Preflight Authorized**: This decision authorizes the execution of the zero-update preflight checks detailed in `PREFLIGHT_CHECKLIST.md`. It does NOT authorize unattended background training processes or optimizer execution.
2. **Deterministic Evaluation Mandatory**: Evaluation must be executed using Task Evaluator v0.1 (`chakrview/evaluation/task_evaluator.py`). Conflating EOS termination with task accuracy is strictly prohibited.
3. **Immutable Baseline**: The canonical Step 10,000 checkpoint (`96357705...`), tokenizer files, and official Exp17 datasets remain completely immutable. All Exp18 checkpoints and outputs must be written solely to `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/`.
