# ChakrView Exp16 (v2) — Automated Test & Verification Report

**Date**: 2026-10-22  
**Test Suite**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py`  
**Execution Command**: `.venv\Scripts\pytest.exe experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py -v`  
**Exit Code**: `0` (All 8 tests passed)  

---

## 1. Test Execution Evidence

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- D:\Project\ChakrView\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Project\ChakrView
configfile: pytest.ini
collecting ... collected 8 items

experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_canonical_and_historical_artifacts_immutability PASSED [ 12%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_original_exp16_datasets_immutability PASSED [ 25%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_repaired_split_counts_and_balance PASSED [ 37%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_zero_cross_split_prompt_duplicates PASSED [ 50%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_target_distribution_randomization PASSED [ 62%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_span_length_contract_compliance PASSED [ 75%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_fail_closed_span_decoding PASSED [ 87%]
experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/test_exp16_repaired.py::test_deterministic_regeneration PASSED [100%]

============================== 8 passed in 1.94s ==============================
```

---

## 2. Test Verification Matrix

| Test Function | Verification Objective | Result |
| :--- | :--- | :---: |
| `test_canonical_and_historical_artifacts_immutability` | Byte-for-byte SHA-256 match for canonical checkpoint, vocab, merges, session.py, and Exp15 checkpoint. | **PASSED** |
| `test_original_exp16_datasets_immutability` | Confirms historical Exp16 v1 datasets in `exp16_pointer_retraining_design/data/` remain untouched. | **PASSED** |
| `test_repaired_split_counts_and_balance` | Asserts exactly 300 Train, 60 Val, 120 Test, balanced across 10 categories. | **PASSED** |
| `test_zero_cross_split_prompt_duplicates` | Audits all pairs across Train, Val, and Test to prove 0 exact prompt duplicates. | **PASSED** |
| `test_target_distribution_randomization` | Proves dynamic binding targets vary across slots 0, 1, 2; distractor states span 5 states; anaphora balances Alpha & Beta. | **PASSED** |
| `test_span_length_contract_compliance` | Asserts $1 \le \text{span\_length} \le 12$ tokens across all 282 answerable examples. | **PASSED** |
| `test_fail_closed_span_decoding` | Proves every answerable example decodes back to the exact target string under canonical tokenizer. | **PASSED** |
| `test_deterministic_regeneration` | Re-verifies file hashes match `EXP16_REPAIR_MANIFEST.json` under fixed seeds. | **PASSED** |
