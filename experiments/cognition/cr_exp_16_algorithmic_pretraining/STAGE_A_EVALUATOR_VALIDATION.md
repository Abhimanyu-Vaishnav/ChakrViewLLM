# CR-EXP-16 Stage A Evaluator Validation
## Synthetic Calibration Suite for Sequence and Token Copying

**Mission Identifier:** CR-EXP-16  
**Calibration Script:** `scripts/evaluator_validation_stage_a.py`  
**Standard:** 10 Known-Answer Edge-Case Unit Tests  

---

## 1. Unit Test Matrix & Verification

| Test ID | Edge-Case Scenario Description | Expected Target | Evaluated Text | Exact Match Flag | Token Accuracy | Status |
|---|---|---|---|---|---|---|
| **CASE_1** | Exact sequence copying | `alpha beta gamma` | `alpha beta gamma` | `True` | 1.00 | **PASS** |
| **CASE_2** | Single-token copying | `Dev` | `Dev` | `True` | 1.00 | **PASS** |
| **CASE_3** | Multi-token copying | `Mukesh` | `Mukesh` | `True` | 1.00 | **PASS** |
| **CASE_4** | Repeated tokens | `red red blue red` | `red red blue red` | `True` | 1.00 | **PASS** |
| **CASE_5** | Long sequences (8 tokens) | `one two three four five six seven eight` | `one two three four five six seven eight` | `True` | 1.00 | **PASS** |
| **CASE_6** | Missing tokens (partial match) | `alpha beta gamma delta` | `alpha beta` | `False` | 0.50 | **PASS** |
| **CASE_7** | Extra tokens appended | `alpha beta` | `alpha beta gamma delta` | `False` | 0.50 | **PASS** |
| **CASE_8** | Incorrect ordering | `alpha beta gamma` | `gamma beta alpha` | `False` | 0.33 | **PASS** |
| **CASE_9** | Early termination / empty output | `alpha beta gamma` | `""` | `False` | 0.00 | **PASS** |
| **CASE_10**| Trailing punctuation / special-token | `alpha beta gamma` | `alpha beta gamma.` | `False` | 0.67 | **PASS** |

**Calibration Verdict:** **ALL 10 UNIT TESTS PASSED.** The evaluator strictly rejects partial, out-of-order, or extra tokens for exact-match credit, while computing exact token alignment.

