# UCIP-09R: Independent Evaluation & Multi-Benchmark Scorecard

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Sealed 500-Probe Benchmark Comparison

Evaluated on `sealed_language_intelligence_500.json` (SHA-256: `3958d3839c0e8ef38fa056cffb24594243cbbe59623dd7460a340a0189981eba`):

| Capability Domain | Probes | UCIP-08 Medium | UCIP-09 Medium | UCIP-09R Candidate | Primary Behavior |
|---|---|---|---|---|---|
| **English conversation** | 70 | 19 / 70 (27.1%) | **22 / 70 (31.4%)** | 5 / 70 (7.1%) | UCIP-09 retains strongest conversational identity |
| **English explanations** | 70 | 1 / 70 (1.4%) | 0 / 70 (0.0%) | 0 / 70 (0.0%) | 12.9M parametric memory limit |
| **Devanagari Hindi** | 70 | 25 / 70 (35.7%) | 27 / 70 (38.6%) | **28 / 70 (40.0%)** | **Strongest Devanagari Hindi pass rate** |
| **Roman Hindi** | 50 | 5 / 50 (10.0%) | **9 / 50 (18.0%)** | 2 / 50 (4.0%) | Conversational colloquial variation |
| **Hinglish** | 70 | 12 / 70 (17.1%) | 12 / 70 (17.1%) | **13 / 70 (18.6%)** | **Highest Hinglish score** |
| **Exact copying** | 40 | 0 / 40 (0.0%) | 0 / 40 (0.0%) | 0 / 40 (0.0%) | Zero-shot verbatim limit |
| **Elementary arithmetic** | 40 | 2 / 40 (5.0%) | **3 / 40 (7.5%)** | 1 / 40 (2.5%) | Arithmetic requires tool integration |
| **Instruction following** | 40 | 0 / 40 (0.0%) | **3 / 40 (7.5%)** | 0 / 40 (0.0%) | Formatting constraint variation |
| **Semantic consistency** | 25 | 8 / 25 (32.0%) | 5 / 25 (20.0%) | **8 / 25 (32.0%)** | Tied for highest constraint adherence |
| **Multi-turn context** | 25 | 0 / 25 (0.0%) | 0 / 25 (0.0%) | 0 / 25 (0.0%) | Requires hybrid memory engine |
| **OVERALL ACCURACY** | **500** | **72 / 500 (14.4%)** | **81 / 500 (16.2%)** | **57 / 500 (11.4%)** | **UCIP-09 remains the top-performing candidate** |

---

## 2. Frozen Owner Reality 100 Benchmark Comparison

Evaluated on `sealed_owner_reality_100.json` (SHA-256: `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15`):
- **UCIP-06 Medium:** 9 / 100 (9.0%)
- **UCIP-07 Medium:** 12 / 100 (12.0%)
- **UCIP-08 Medium:** 10 / 100 (10.0%)
- **UCIP-09 Medium Candidate:** **10 / 100 (10.0%)**
- **UCIP-09R Candidate:** **8 / 100 (8.0%)**

---

## 3. Analysis & Key Finding

Training on compact natural arithmetic and copying datasets (`ucip09r_train.jsonl`) succeeded on targeted synthetic tasks (as proven in the minimal learning proof), but caused catastrophic forgetting of general English conversational greeting tone on the broader 500-probe suite (dropping English conversation from 31.4% to 7.1%).

**UCIP-09 Medium Candidate (`5513fefe...`) remains the superior, balanced release artifact**, achieving the highest overall accuracy (**16.2% on 500 probes**) and the most fluent identity articulation.
