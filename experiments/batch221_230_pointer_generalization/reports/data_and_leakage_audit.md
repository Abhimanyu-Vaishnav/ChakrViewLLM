# Batch 222 & 224: Data and Leakage Audit for Batches 221–230

## 1. Frozen Data Lineage
- **Corpus Origin**: `experiments/batch211_220_relational_learning/data/`
- **Total Corpus Size**: 10,000 synthetic records.
- **Split Breakdown**:
  - Train: 6,360 records (25 entities, 50 relations, 100 facts)
  - Validation: 1,097 records (25 entities, 50 relations, 100 facts)
  - Test Pool: 2,543 records (9 entities, 24 relations, 36 facts)
- **Primary Frozen Benchmark**: `experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json` (100 records, SHA-256: `f345670a7b49ed1fd17b9c700fbcf468e597ee3035e50278dc439dbfdc0fd7aa`).

---

## 2. Leakage and Overlap Audit
1. **Entity Leakage**:
   - Training Entities: 25 distinct names.
   - Held-Out Test Entities: 9 distinct names (`Copper`, `Euler number e`, `Gold`, `Golden Ratio phi`, `Hash Table`, `Rust`, `Saturn`, `चांदी (Silver)`, `तांबा (Copper)`).
   - Intersection Count: **0 entities** ($0.0\%$ leakage).
2. **Fact Tuple Leakage**:
   - Triple $(E, R, V)$ Intersection Count between Train and Test: **0 triples** ($0.0\%$ leakage).
3. **Template Family Overlap**:
   - Context template formats are shared between train and test to measure structural generalization, but exact lexical spans for test entities never appear in the training corpus.
4. **Oracle Candidate Check**:
   - Candidates for TokenTrie are generated exclusively from context words and n-grams ($n \le 3$) of each individual example. No test labels or global vocabulary lookups are permitted.
