# ChakrView Overnight Mission — Test Results

**Mission Directory**: `experiments/overnight_runs/mission_20261023_003512/`  
**Execution Timestamp**: 2026-10-23T00:36:30+05:30  
**Status**: **ALL TESTS PASSED (OK)**

---

## 1. Unit Test Suite Execution

### Command Executed
```powershell
.venv\Scripts\python.exe -m unittest discover -s experiments/research_v09_conversation_sft/exp17_instruction_sft_v1 -p "test_*.py"
```

### Execution Output & Metrics
- **Tests Discovered & Run**: 9
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: 5.057s
- **Exit Code**: `0`

### Test Case Breakdown
1. `test_bank_split_counts`: Verified exact split counts across candidate banks (Train: 40, Val: 5, Test: 5).
2. `test_bank_schema_and_types`: Verified tuple structures, prompt-response field types, and metadata tags.
3. `test_factual_qa_disjointness`: Verified 100% disjoint entities, questions, and responses between Train, Val, and Test splits.
4. `test_explanation_diversity`: Verified 13 distinct physical and causal mechanisms across 50 explanation items.
5. `test_abstention_absence_proof`: Verified absence of requested attributes across all 38 unanswerable prompts.
6. `test_control_grounding_proof`: Verified that all 12 answerable control prompts have ground-truth answers in their context.
7. `test_generator_in_memory_integration`: Verified that `Exp17DatasetGenerator().generate_all_splits()` runs deterministically.
8. `test_normal_chat_gate_isolation`: Verified zero exact or normalized prompt leakage against all 20 frozen Normal Chat Gate v0.1 prompts.
9. `test_response_token_bounds`: Verified that all responses lie strictly within the 3–45 token range (target 5–35).

---

## 2. In-Memory Generation Preflight Check

### Command Executed
```powershell
.venv\Scripts\python.exe experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/generate_exp17_datasets.py
```

### Execution Output & Metrics
- **Verification Mode**: Dry-run in-memory preflight (0 files written).
- **Split Distribution**:
  - `train`: 320 records (8 categories x 40 items)
  - `val`: 40 records (8 categories x 5 items)
  - `test`: 40 records (8 categories x 5 items)
- **Token Bounds**:
  - Min tokens: 6
  - Max tokens: 34
  - Mean tokens: 17.82
  - Target 5–35 compliance: 100.0%
- **Contamination & Gate v0.1 Leakage**: 0 matches detected.
- **Exit Code**: `0`
