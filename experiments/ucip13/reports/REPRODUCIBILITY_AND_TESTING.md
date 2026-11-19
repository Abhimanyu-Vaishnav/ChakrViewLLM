# UCIP-13: Reproducibility & Testing Report

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Test Suite:** `experiments/ucip13/tests/test_ucip13_training.py`  
**Execution Environment:** Windows, Python 3.14.7, pytest 9.1.1  

---

## 1. Automated Verification Suite Execution

```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- D:\Project\ChakrView\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Project\ChakrView
configfile: pytest.ini
collecting ... collected 6 items

experiments/ucip13/tests/test_ucip13_training.py::test_01_all_checkpoints_exist PASSED [ 16%]
experiments/ucip13/tests/test_ucip13_training.py::test_02_checkpoint_hashes_unique PASSED [ 33%]
experiments/ucip13/tests/test_ucip13_training.py::test_03_training_loss_progression PASSED [ 50%]
experiments/ucip13/tests/test_ucip13_training.py::test_04_validation_loss_reduction PASSED [ 66%]
experiments/ucip13/tests/test_ucip13_training.py::test_05_held_out_test_results PASSED [ 83%]
experiments/ucip13/tests/test_ucip13_training.py::test_06_protected_session_hash PASSED [100%]

============================== 6 passed in 1.84s ==============================
```

---

## 2. Invariant Protections Verified

- `chakrview/runtime/session.py` hash verified identical to `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`.
- Zero training executed on held-out test shards.
- Checkpoints saved in non-destructive append-only structure.
