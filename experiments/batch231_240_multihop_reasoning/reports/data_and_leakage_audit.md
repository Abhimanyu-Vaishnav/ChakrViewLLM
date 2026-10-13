# Batch 233: Multi-Hop Data and Leakage Audit

## 1. Corpus Statistics
- **Total Generated Chains**: 1,950 records across 25 foundational chains in Science, Chemistry, Mathematics, Coding, and Multilingual Hindi (Devanagari).
- **Split Partitioning by Base Entity $E_1$**:
  - **Train**: 1,200 records (15 distinct $E_1$ base entities).
  - **Val**: 200 records (4 distinct $E_1$ base entities).
  - **Test**: 550 records (6 distinct held-out $E_1$ base entities: `Saturn`, `Table Salt`, `Red-Black Tree`, `Square`, `कांस्य (Bronze)`, `बृहस्पति (Jupiter)`).
- **Primary Frozen Benchmark**: `experiments/batch231_240_multihop_reasoning/data/frozen_multihop_manifest_100.json` (100 records, SHA-256: `b70b38a880e307be77d5eaa1df68bd86b977ad67a2f9d13a79e4b1b7e0939016`).

---

## 2. Zero-Leakage Verification
1. **Entity Leakage**:
   - Training base entities: 15.
   - Test base entities: 6 (`Saturn`, `Table Salt`, `Red-Black Tree`, `Square`, `कांस्य (Bronze)`, `बृहस्पति (Jupiter)`).
   - Overlap: **0 entities** ($0.0\%$ leakage).
2. **Intermediate Link & Target Value Leakage**:
   - Zero test chain triples $(E_1, P_1, E_2, R_2, V^*)$ appear in the training corpus.
3. **Candidate Space**:
   - All candidate spans are bounded strictly by words and n-grams ($n \le 3$) of the supplied multi-document context for each individual evaluation record.
