# CHAKRVIEW — EXP35 DATASET QUALITY & INTEGRITY REPORT
## Comprehensive Audit of the 1,000-Record DPCE Instruction Mixture

**Dataset:** `exp35_full_dataset.jsonl` (SHA-256 `ac247d1e9a61b945640b094d3dda63a355fba704d362820d2225ebdac20b8c8d`)  
**Train Split:** `exp35_train.jsonl` (900 records, SHA-256 `5b10b7d58e891e3cadb1213f436417dd34701b6ed3e0573ba2d534a786e7ad1d`)  
**Val Split:** `exp35_validation.jsonl` (100 records, SHA-256 `a30d633f8006ae1175cda8d3277c7b94f123665dfaf555f54263deace68f1a6a`)  
**Manifests:** `exp35_dataset_manifest.json`, `exp35_split_manifest.json`  
**Date:** 2026-10-30  

---

## 1. DATASET STATISTICAL QUALITY METRICS

- **Total Records:** Exactly 1,000.
- **Split Ratio:** Exactly 900 Training (90.0%) / 100 Validation (10.0%).
- **Unique Prompts:** 1,000 / 1,000 (100% unique prompt strings).
- **Unique Responses:** **1,000 / 1,000 (100% unique response strings)**.
- **Response Diversity Ratio:** **100.0%** (resolving M4 Finding D1).
- **Context Boundaries:** Max total sequence length is 176 tokens (mean: 97.9 tokens), perfectly bounded within the 512 context limit.
- **Review Status:** All records marked `VERIFIED_TRAINING` with synthetic provenance.

### Category Breakdown
1. `single_entity_extraction`: 150 records (135 train / 15 val)
2. `attribute_extraction`: 150 records (135 train / 15 val)
3. `multi_entity_disambiguation`: 150 records (135 train / 15 val)
4. `multi_attribute_selection`: 100 records (90 train / 10 val)
5. `slot_filling`: 100 records (90 train / 10 val)
6. `closed_world_question_answering`: 100 records (90 train / 10 val)
7. `context_dependent_extraction`: 100 records (90 train / 10 val)
8. `simple_multi_turn_fact_retention`: 75 records (67 train / 8 val)
9. `changed_fact_tracking`: 75 records (68 train / 7 val)

---

## 2. DATA PROVENANCE & GOVERNANCE

- **Authoring Engine:** Indigenous synthetic procedural generation script (`seed=42`).
- **External Dependencies:** Zero external LLM APIs, zero pre-existing proprietary datasets.
- **Entity Pool:** Non-personal synthetic entity pools (controlled fictional agents, cities, and project names).
- **License Designation:** CC0-1.0 (Public Domain Dedication) for project-generated synthetic text.

