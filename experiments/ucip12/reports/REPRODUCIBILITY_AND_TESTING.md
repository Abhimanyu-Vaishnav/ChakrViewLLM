# UCIP-12: Reproducibility, Verification & Testing Report

**Mission:** UCIP-12 — Large-Scale Corpus Foundation & Learning Readiness  
**Test Suite:** `experiments/ucip12/tests/test_ucip12_corpus.py`  
**Execution Environment:** Windows, Python 3.14.7, pytest 9.1.1  

---

## 1. Test Suite Verification

An automated verification test suite was established and executed via `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Project\ChakrView
configfile: pytest.ini
collected 6 items

experiments/ucip12/tests/test_ucip12_corpus.py::test_protected_session_hash PASSED      [ 16%]
experiments/ucip12/tests/test_ucip12_corpus.py::test_frozen_tokenizer_roundtrip PASSED  [ 33%]
experiments/ucip12/tests/test_ucip12_corpus.py::test_stage_c_manifest_and_scale PASSED [ 50%]
experiments/ucip12/tests/test_ucip12_corpus.py::test_stage_c_binary_shards_exist PASSED[ 66%]
experiments/ucip12/tests/test_ucip12_corpus.py::test_sealed_benchmark_integrity PASSED  [ 83%]
experiments/ucip12/tests/test_ucip12_corpus.py::test_source_registry_cleanliness PASSED [100%]

============================== 6 passed in 2.45s ==============================
```

---

## 2. Cryptographic Integrity Checklist

All verified components and hashes:
1. `chakrview/runtime/session.py`:
   `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (MATCH)
2. `experiments/ucip12/benchmarks/sealed/benchmark_r2_sealed.json`:
   `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` (MATCH)
3. `data/experiments/vocab_4096/tokenizer_config.json`:
   Merges SHA-256: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f` (MATCH)
4. `data/manifests/stage_c_manifest.json`:
   SHA-256: `6c744118f0fc93c24cf425f9613644306c8b010c33d01f0421c157bcadb49296` (MATCH)

---

## 3. Reproduction Instructions

See `experiments/ucip12/REPRODUCE.md` for one-command re-verification of all hashes, fertility metrics, and test assertions.
