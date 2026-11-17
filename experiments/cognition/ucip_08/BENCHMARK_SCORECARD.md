# UCIP-08: Benchmark Scorecard (Independent Reality Verification)

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Benchmark:** Sealed Owner Reality 100-Probe Evaluation Suite (`sealed_owner_reality_100.json`)  
**SHA-256 Digest of Benchmark:** `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15`  
**Date:** November 17, 2026  

---

## 1. Comparative Performance Matrix

| Capability Domain | Probe Count | UCIP-06 Medium | UCIP-07 Medium | UCIP-08 Candidate | Net Capability Trend |
|---|---|---|---|---|---|
| **English Conversation** | 15 | 2 / 15 (13.3%) | **4 / 15 (26.7%)** | 2 / 15 (13.3%) | Baseline conversational competence |
| **English Explanations** | 10 | 1 / 10 (10.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Unreliable zero-shot factual reasoning |
| **Devanagari Hindi** | 15 | 2 / 15 (13.3%) | **5 / 15 (33.3%)** | 4 / 15 (26.7%) | Strongest non-English domain |
| **Roman Hindi** | 10 | 1 / 10 (10.0%) | 1 / 10 (10.0%) | 1 / 10 (10.0%) | Stable basic template completion |
| **Hinglish** | 15 | 2 / 15 (13.3%) | 1 / 15 (6.7%) | 2 / 15 (13.3%) | Partial vocabulary associations |
| **Exact Copying** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Fragile instruction adherence |
| **Elementary Arithmetic** | 10 | 0 / 10 (0.0%) | 1 / 10 (10.0%) | 1 / 10 (10.0%) | Isolated computation hits |
| **Instruction Following** | 10 | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | Structural constraints unlearned |
| **Semantic Consistency** | 5 | 1 / 5 (20.0%) | 0 / 5 (0.0%) | 0 / 5 (0.0%) | Constrained reasoning unlearned |
| **OVERALL ACCURACY** | **100** | **9 / 100 (9.0%)** | **12 / 100 (12.0%)** | **10 / 100 (10.0%)** | **Real 12.9M zero-shot ceiling: ~10–12%** |

---

## 2. Benchmark Artifact Discrepancy Reconciliation

### Reported Benchmark vs Calibrated Independent Reality

| Model Candidate | Prior Reported Benchmark (Permuted Stems) | Calibrated Independent Benchmark (Sealed Reality) | Discrepancy Magnitude |
|---|---|---|---|
| **UCIP-06 Medium** | 26.50% (212 / 800) | **9.00% (9 / 100)** | $-17.50\%$ (Artificial Inflation) |
| **UCIP-07 Medium** | 31.37% (251 / 800) | **12.00% (12 / 100)** | $-19.37\%$ (Artificial Inflation) |
| **UCIP-08 Candidate** | N/A | **10.00% (10 / 100)** | Calibrated Truth Baseline |

### Explanation of Discrepancy
1. **Permuted Stem Contamination:** The prior 800-probe suite repeated 80 base stems 10 times across trivial prompt prefix variants. 47 of those stems were present in training data.
2. **Evaluator False Positives:** The prior regex evaluator accepted unigram substring hits (e.g., `'7'` inside `'17'`) and unigram keywords (`'water'` in photosynthesis), which awarded passing credit to 119 failed outputs.
3. **Decoupled Verification:** When tested on 100 held-out probes with calibrated integer extraction and repetition rejection, true zero-shot capability across all 12.9M candidates sits between **9% and 12%**.
