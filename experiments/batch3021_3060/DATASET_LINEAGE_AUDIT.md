# Dataset Lineage & Provenance Audit — Batch 3021–3060
## End-to-End Tracking from Raw Corpora and Authored Instruction Seeds

### 1. Executive Summary

This audit establishes the explicit lineage, derivation history, template family relationships, and factual origins for every example in the ChakrView balanced SFT dataset mixture.

The dataset contains **400 total records**:
- **280 Instruction Records (70.0%)**: Derived across 7 verified instruction categories (40 per category).
- **120 Base-Continuation Replay Records (30.0%)**: Sourced directly from verified raw pretraining corpora.

---

### 2. Instruction Data Lineage & Ancestry

Every instruction record originated from two sequential authoring phases:

| Phase | Batches | Record Count | Categories Covered | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Batch 2821–2860 | 210 records | 7 categories (30 records each) | Authored to explore instruction adherence; cases 01..24 were train, 25..27 val, 28..30 test in Batch 2821. |
| **Phase 2** | Batch 2941–2980 | 70 records | 7 categories (10 records each) | Extended categories to 40 items each (items 31..40 in each category: 2 train, 4 val, 4 test). |

#### Lineage per Category (40 Records Each):
1. **`simple_instruction`** (`INST_001` .. `INST_040`):
   - Items 01..30: Authored in Batch 2821 (`experiments/batch2821_2860/build_curated_dataset.py`).
   - Items 31..40: Authored in Batch 2941 (`experiments/batch2941_2980/build_balanced_dataset.py`).
   - Template Family: Single-turn factual directives without context block (`"Instruction: {directive}\nResponse:"`).
   - Factual Independence: 40 distinct domain facts across computer science, physics, networking, and biology.
2. **`context_extraction`** (`EXTR_001` .. `EXTR_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: Single-entity extraction (`"Instruction: Extract the requested detail from the text:\n{context}\nResponse:"`).
   - Factual Independence: 40 unique entity-context snippets (UUIDs, ports, filenames, telemetry metrics).
3. **`context_grounded_qa`** (`CQA_001` .. `CQA_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: Question answering conditioned on an explicit paragraph (`"Instruction: Answer the question using only the provided context.\nContext: {context}\nQuestion: {question}\nResponse:"`).
4. **`short_summarization`** (`SUMM_001` .. `SUMM_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: One-sentence summaries of 2-3 sentence technical descriptions.
5. **`structured_output`** (`STR_001` .. `STR_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: Strict JSON key-value formatting without markdown wrappers.
6. **`uncertainty_insufficient_info`** (`UNC_001` .. `UNC_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: Abstention testing where context is missing the requested answer.
7. **`bounded_text_transformation`** (`TRNS_001` .. `TRNS_040`):
   - Items 01..30: Authored in Batch 2821.
   - Items 31..40: Authored in Batch 2941.
   - Template Family: Deterministic string operations (casing, reversing, bracket wrapping).

---

### 3. Replay Data Lineage & Corpus Provenance

The 120 replay examples are extracted directly from pretraining corpus files located in `data/raw/`:
- `data/raw/english/english_computer_science.txt` (`corpus_english_cs`, 12 lines)
- `data/raw/english/english_systems.txt` (`corpus_english_systems`, 15 lines)
- `data/raw/english/english_foundation.txt` (`corpus_english_foundation`, 73 lines)
- `data/raw/code/code_python.txt` (`corpus_code_python`, 8 lines)
- `data/raw/code/code_foundation.txt` (`corpus_code_foundation`, 25 lines)
- `data/raw/mathematics/math_foundation.txt` (`corpus_math_foundation`, 27 lines)
- `data/raw/english/stage_b_systems.txt` (`corpus_stage_b_systems`, 1139 lines)

Every replay record is a natural continuation sequence split at the approximate midpoint of an authentic sentence (`prompt` = first ~50%, `response` = second ~50%). Zero instruction templates or conversational wrappers are applied.

---

### 4. Split Overlap & Leakage Status

- **Exact Duplicate Prompts:** 0 (Disjoint across splits).
- **Normalized Duplicate Prompts:** 0 (Disjoint across splits).
- **Duplicate Response Targets:** 0 across all 280 instruction records (all 280 responses are unique).
- **Historical Benchmark Overlap:** 21 records from the historical 24-case benchmark (`experiments/batch2821_2860/SFT_EVALUATION_SET.json`) are present in `train.jsonl` (cases 28..30 of 7 categories).
- **Source-Document Partitioning Defect:** In Batch 2941–2980, sequential accumulation allowed `corpus_english_foundation` to span `train` and `val`, and `corpus_code_foundation` to span `val` and `test`. Corrected in Phase B via document-level pre-split assignment.
