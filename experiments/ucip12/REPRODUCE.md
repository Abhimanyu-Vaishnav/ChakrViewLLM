# UCIP-12 Reproduction Guide

This guide details how to reproduce all forensic findings, scale measurements, tokenizer fertility audits, and automated test validations for the UCIP-12 research mission.

---

## 1. Environment Verification

Ensure Python 3.12+ / 3.14+ is active with the project dependencies installed:
```powershell
.venv\Scripts\python.exe --version
```

---

## 2. Execute Automated Verification Suite

Run the full pytest suite:
```powershell
.venv\Scripts\pytest.exe experiments/ucip12/tests/test_ucip12_corpus.py -v
```
Expected result: **6 passed in < 3 seconds**.

---

## 3. Re-run Tokenizer Fertility Audit

Execute the standalone multi-domain fertility measurement tool:
```powershell
.venv\Scripts\python.exe experiments/ucip12/scripts/audit_tokenizer_fertility.py
```
Outputs `experiments/ucip12/manifests/tokenizer_fertility_audit.json` detailing exact tokens/word and lossless roundtrip validation.

---

## 4. Re-run Source Registry Verification

Audit source registry licenses and approved/rejected flags:
```powershell
.venv\Scripts\python.exe experiments/ucip12/scripts/audit_sources_json.py
```

---

## 5. Verify Cryptographic Hashes

To verify critical hashes independently:
```powershell
.venv\Scripts\python.exe -c "
import hashlib
for p in ['chakrview/runtime/session.py', 'experiments/ucip12/benchmarks/sealed/benchmark_r2_sealed.json']:
    print(p, hashlib.sha256(open(p, 'rb').read()).hexdigest())
"
```
Expected:
- `session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- `benchmark_r2_sealed.json`: `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef`
