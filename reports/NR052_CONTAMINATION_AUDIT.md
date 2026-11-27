# NR-052: Benchmark Contamination Audit Report

**Mission:** CHAKRVIEW NR-052  
**Date:** November 27, 2026  
**Status:** AUDIT PASSED — 0.0% CONTAMINATION

---

## 1. Scope & Test Methodology

We evaluated all training candidates against the two canonical sealed benchmark sets:
1. **Sealed 120-Prompt Multi-Domain Benchmark (`sealed_120_benchmark.json`)**:
   - 120 held-out prompts spanning English, Hindi, Hinglish, Logic, Context, QA, and Correction.
2. **Fresh Matched 1,000 Benchmark (`fresh_matched_benchmark_1000.json`)**:
   - 1,000 held-out evaluation tasks established in NR-020.

The audit checked for:
- Exact case-insensitive prompt substring matches.
- Exact answer/expected keyword leakage.
- Normalized 10-gram phrase overlap.

---

## 2. Contamination Results by Source

| Data Candidate | Format | Total Records | Exact Prompt Matches | 10-gram Overlap Count | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage-B Raw Foundation** (`data/raw/`) | TXT | 27 files | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |
| **Corpus V3 Train** (`data/corpus_v3/`) | JSONL | 4,075 | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |
| **Stage-D Train** (`data/stage_d/`) | JSONL | 4,686 | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |
| **Stage-E Train** (`data/stage_e_corpus_013/`) | JSONL | 5,238 | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |
| **SFT UCIP14R** (`data/sft/ucip14r/`) | JSONL | 15,000 | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |
| **SFT UCIP17** (`data/sft/ucip17/`) | JSONL | 15,000 | 0 / 120 (0.0%) | 0 (0.0%) | **CLEAN** |

### Conclusion
Zero contamination was detected in any raw foundation prose or conversational training split. All benchmark test prompts remain strictly unseen out-of-distribution evaluation targets.
