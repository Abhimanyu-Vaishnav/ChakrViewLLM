# UCIP-09: Regression & Verification Suite Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Test Execution Summary

The complete project regression test suite was executed using:

```powershell
.\.venv\Scripts\pytest.exe tests/test_architecture_loader.py tests/test_inference_session.py tests/test_ucip07r_loader.py
```

### Results Matrix:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Project\ChakrView
configfile: pytest.ini
collected 22 items

tests\test_architecture_loader.py ......                                 [ 27%]
tests\test_inference_session.py .......                                  [ 59%]
tests\test_ucip07r_loader.py .........                                   [100%]

============================= 22 passed in 10.09s =============================
```

- **Total Test Cases Collected:** 22
- **Passed:** 22
- **Failed:** 0
- **Skipped / Timed Out:** 0
- **Pass Rate:** **100.0%**

---

## 2. Invariant & Lineage Verification

1. **Protected File:** `chakrview/runtime/session.py` remains 100% bit-exact with SHA-256 `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`.
2. **Canonical Baseline Initialization:** `ChakrMicro(seed=42)` computed hash remains `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
3. **Canonical Stage B:** Hash remains `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
4. **Frozen Owner Reality 100 Benchmark:** Embedded `sha256` remains `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15`.
5. **Sealed Language Intelligence 500 Benchmark:** Embedded `sha256` sealed at `3958d3839c0e8ef38fa056cffb24594243cbbe59623dd7460a340a0189981eba`.
