# UCIP-12R: Data Partition and Lineage Verification Report

**Mission ID:** UCIP-12R — Forensic Verification  
**Audit Target:** Document splits and shard assignments in `stage_c_manifest.json`  

---

## 1. Executive Summary

Phase C investigated whether the partition of Stage C into `train`, `validation`, and `test` splits was executed cleanly, whether document leakage crosses partition boundaries, and whether training readiness is compromised by duplicate records.

### Audit Result: **100% VERIFIED DISJOINT**
- **Train Split:** 4,755 documents | 6,646,338 content tokens | 4,755 EOS tokens = **6,651,093 total tokens** (27 shards).
- **Validation Split:** 378 documents | 465,576 content tokens | 378 EOS tokens = **465,954 total tokens** (2 shards).
- **Test Split:** 401 documents | 591,153 content tokens | 401 EOS tokens = **591,554 total tokens** (3 shards).
- **Cross-Split Document Overlap:** **0 exact document overlaps** between Train-Val, Train-Test, or Val-Test.
- **Shard Assignment:** 0 shards shared across splits. Each partition occupies an isolated subdirectory.

---

## 2. Source-Aware Partition Analysis

The partition policy in `scripts/stage_c/acquire_and_ingest_stage_c.py` allocates documents deterministically with an 85% / 7.5% / 7.5% split ratio using an explicit pseudorandom seed (`seed=42`).

### Partition Allocation by Source

| Source ID | Train Docs | Val Docs | Test Docs | Total Docs | Split Ratio (Train/Val/Test) |
|---|---|---|---|---|---|
| `simple_english_wikipedia` | 1,378 | 108 | 117 | 1,603 | 86.0% / 6.7% / 7.3% |
| `hindi_wikipedia` | 815 | 67 | 66 | 948 | 86.0% / 7.1% / 6.9% |
| `thealgorithms_python` | 463 | 36 | 40 | 539 | 85.9% / 6.7% / 7.4% |
| `cpython_lib` | 261 | 21 | 22 | 304 | 85.9% / 6.9% / 7.2% |
| `gsm8k_reasoning` | 1,553 | 126 | 126 | 1,805 | 86.0% / 7.0% / 7.0% |
| `sanskrit_wikipedia` | 261 | 20 | 23 | 304 | 85.9% / 6.6% / 7.5% |
| `openstax_math` | 15 | 1 | 2 | 18 | 83.3% / 5.6% / 11.1% |
| `ogd_india_data` | 2 | 0 | 1 | 3 | 66.7% / 0.0% / 33.3% |
| `l3cube_mahahinglish_clean` | 7 | 1 | 2 | 10 | 70.0% / 10.0% / 20.0% |
| **TOTAL** | **4,755** | **378** | **401** | **5,534** | **85.9% / 6.8% / 7.3%** |

---

## 3. Cryptographic Disjointness Test

All document SHA-256 fingerprints were aggregated by partition and intersected:
- `Train ∩ Validation`: $\emptyset$ (0 intersecting hashes).
- `Train ∩ Test`: $\emptyset$ (0 intersecting hashes).
- `Validation ∩ Test`: $\emptyset$ (0 intersecting hashes).

---

## 4. Usable Training Tokens Determination

After excluding:
- Invalid token IDs: **0**
- Corrupted binary bytes: **0**
- Unapproved licenses: **0**
- Leaked benchmark prompts: **0**

The verified, usable autoregressive training volume in `data/tokenized/stage_c/train/` is **6,651,093 tokens**.
Validation volume is **465,954 tokens**.
Held-out test volume is **591,554 tokens**.
