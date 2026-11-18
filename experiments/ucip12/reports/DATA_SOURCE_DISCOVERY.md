# UCIP-12: Data Source Discovery & Ingestion Assessment

**Mission:** UCIP-12 — Large-Scale Corpus Foundation & Learning Readiness  
**Author:** Antigravity  

---

## 1. Context and Problem Statement

In UCIP-11, ChakrView models (Condition A through Condition E) achieved near 0% accuracy across unseen sealed benchmark probes. The immediate hypothesis posited in UCIP-11 was:
> *"The available training data is inadequate for reliable language acquisition... the available clean pretraining corpus contained fewer than 100,000 tokens."*

The goal of this investigation was to perform exhaustive source discovery to determine what large-scale data exists or can be reliably ingested into ChakrView.

---

## 2. Source Discovery Findings

During source discovery across the repository, the following major repositories of text data were identified:

1. **Stage C Ingestion Pipeline (`scripts/stage_c/acquire_and_ingest_stage_c.py`):**
   - Engineered in Step 6.5 to acquire, clean, deduplicate, and shard text across 8 distinct knowledge and language domains.
   - Total acquired tokens: **7,703,067 tokens**.
   - Sharded format: Little-endian uint16 binary files compatible with `chakrview.training.sharding.read_shard_tokens`.

2. **Stage B Corpus (`data/raw/` and `data/processed/`):**
   - Curated conversational, mathematical, and code snippets (~66k tokens).
   - Serves as targeted alignment data rather than foundational pretraining data.

3. **Curriculum & Arena Sets (`data/tokenized/`):**
   - `coding_arena`: 43k tokens
   - `reasoning_step54`: 52k tokens
   - `curriculum_step53`: 120k tokens

---

## 3. Gap Analysis: Why Did UCIP-11 Not Benefit from Stage C?

A retrospective audit of `experiments/ucip11/scripts/` revealed the critical operational disconnect:
- The pretraining scripts for UCIP-11 Condition B (`Condition_B_CPT`) were hardcoded to point to a local directory `data/tokenized/stage_b/` instead of `data/tokenized/stage_c/`.
- As a result, the training run completed in only a few hundred steps and plateaued because the dataset was exhausted in less than 3 epochs.
- The 7.7M-token Stage C shards remained idle on disk throughout the entire UCIP-11 execution.

---

## 4. Ingestion Action Plan for UCIP-13 Scale-Up

1. **Direct Integration of Stage C Shards:**
   - Pretraining dataloaders must stream directly from `data/tokenized/stage_c/train/` (27 shards, 6.65M tokens).
2. **Validation Shards:**
   - Dataloaders must compute loss against `data/tokenized/stage_c/validation/` (2 shards, 465k tokens) to track true cross-entropy perplexity.
3. **Sequential Curriculum:**
   - Phase 1: Pretraining on Stage C (7.7M tokens) to teach grammar, semantics, multilingual alignment, and code syntax.
   - Phase 2: SFT Fine-Tuning on Exp34 v3 / CBR-SFT (100k tokens) with verified zero-leakage prompt wrappers.
