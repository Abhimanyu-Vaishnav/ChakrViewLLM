# UCIP-12R Reproduction Guide

This guide details how to reproduce all independent forensic findings, shard decodings, tokenizer tests, and automated assertions for the UCIP-12R verification mission.

---

## 1. Run Automated Master Verification Test Suite

Run all 16 automated verification tests:
```powershell
.venv\Scripts\pytest.exe experiments/ucip12r/tests/test_ucip12r_verification.py -v
```
Expected output: **16 passed in ~2 seconds**.

---

## 2. Standalone Verification Scripts

### A. Stage C Shard Audit
Decodes all 32 shards and validates token counts, byte alignment, and ID ranges:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_stage_c_shards.py
```

### B. Corpus Count Reconciliation
Computes exact token counts across all binary shards in the repository:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_corpus_counts.py
```

### C. Tokenizer Independent Verification
Verifies tokenizer loading, lossless roundtrip across 9 categories, and word fertility:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_tokenizer.py
```

### D. Source Registry & Provenance Audit
Audits source licensing and verifies absence of non-commercial or barred sources:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_provenance.py
```

### E. Benchmark Contamination Scan
Checks for prompt overlaps between Stage C, historical SFT, and sealed benchmarks:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_contamination.py
```

### F. UCIP-11 Score Audit
Audits raw evaluation records and verifies that the reported 1.0% pass is a false positive:
```powershell
.venv\Scripts\python.exe experiments/ucip12r/scripts/verify_ucip11_scores.py
```

---

## 3. Verify Protected Hashes

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
