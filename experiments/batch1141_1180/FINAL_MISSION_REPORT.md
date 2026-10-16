# Final Mission Report — Batches 1141–1180
## Corpus-Scale Generalization, Semantic Retrieval Stress Test & Evidence-Span Learning Readiness

### 1. Executive Summary
Batches 1141–1180 scaled ChakrView's local knowledge assistant from a 5-document sample corpus to a 52-document, 116-chunk benchmark across six technical domains. The mission audited the provenance of the indigenous `ChakrBiEncoder` neural model, evaluated embedding caching efficiency, diagnosed and solved the unresolved multi-turn follow-up failure, stress-tested concise evidence span extraction, and performed a data-readiness audit for learned span boundary prediction.

Repository invariants were strictly preserved:
- Canonical training checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`) SHA-256 verified identical before and after.
- `chakrview/runtime/session.py` was untouched; all existing user changes were preserved.
- No third-party LLMs, cloud APIs, or pretrained Hugging Face models were used.

### 2. Retrieval Comparison Table (70-Item Held-Out Suite)

| Metric | Configuration 1 (BM25 Lexical) | Configuration 2 (Hybrid Hash) | Configuration 3 (Hybrid Neural Bi-Encoder) |
| :--- | :--- | :--- | :--- |
| **Retrieval Recall@1** | **93.33% (56/60)** | **93.33% (56/60)** | 91.67% (55/60) |
| **Retrieval Recall@3** | 98.33% (59/60) | 98.33% (59/60) | **100.00% (60/60)** |
| **Retrieval Recall@5** | 98.33% (59/60) | 98.33% (59/60) | **100.00% (60/60)** |
| **Retrieval MRR** | **0.9583** | **0.9583** | 0.9556 |
| **Overall Accuracy** | 84.29% (59/70) | 84.29% (59/70) | 84.29% (59/70) |
| **Mean Token F1** | 0.0832 | 0.0832 | 0.0832 |
| **Citation Validity** | 60/60 (100.0%) | 60/60 (100.0%) | 60/60 (100.0%) |
| **Citation Support** | 54/60 (90.0%) | 54/60 (90.0%) | 54/60 (90.0%) |
| **Query Latency (Median / p95)** | **0.98 ms / 1.22 ms** | 0.97 ms / 1.73 ms | 2.74 ms / 3.95 ms |
| **Ingest Time (116 Chunks)** | **60.8 ms** | 61.5 ms | 373.8 ms (uncached: 7,877 ms) |

### 3. Performance by Question Category (70 Held-Out Queries)

| Category | Query Count | Accuracy | Recall@1 | Mean Token F1 | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **exact_entity** | 10 | 90.0% | 100.0% | 0.0610 | Reliable entity and parameter retrieval. |
| **paraphrase** | 10 | 90.0% | 100.0% | 0.1149 | Robust across altered syntax and sentence structures. |
| **synonym** | 10 | 100.0% | 100.0% | 0.1055 | Handled alternate technical terminology. |
| **distractor_competition**| 10 | 80.0% | 90.0% | 0.0686 | Discriminated identical numbers (`512`, `1536`, `15`) across different domains. |
| **version_conflict** | 6 | 83.3% | 100.0% | 0.1073 | Successfully separated V0.6 features from V0.7 draft deprecations. |
| **follow_up** | 8 | 100.0% | 100.0% | 0.0653 | Fully resolved anaphora across pronouns (`it`, `that tier`). |
| **multilingual_hinglish** | 6 | 83.3% | 83.3% | 0.0543 | Handled mixed Hindi/English queries. |
| **abstention** | 5 | 100.0% | N/A | 0.0000 | Safely abstained on out-of-domain unindexed facts. |
| **false_premise** | 5 | 0.0% | 80.0% | 0.0000 | Assistant answered the underlying factual topic rather than refuting. |

### 4. Root Cause and Resolution of Follow-Up Failure
- **Diagnosis**: The prior failure on `"What symmetric cipher algorithm is used with it?"` was caused by three interacting flaws:
  1. Pronoun matching (`"it"`) was evaluated via raw substring search, causing unrelated words like `"capital"` to trigger follow-up mode.
  2. The candidate scoring formula gave a 5.0x multiplier to clean query tokens while giving retrieval score only 0.5x, causing stopword verbs like `"used"` to outscore the true source chunk.
  3. Context resolution lacked prior entity support.
- **Resolution**: Implemented word-boundary regex matching for follow-up triggers, added auxiliary verbs (`"use"`, `"used"`, `"using"`, `"with"`) to the stopword filter, and elevated retrieval score weight ($8.0 \times c.score$).
- **Empirical Result**: 8/8 (100.0%) held-out follow-up questions passed cleanly.

### 5. Embedding Efficiency & Caching Results
- Added `CachedEmbeddingProvider` with LRU caching, SHA-256 composite keys, and automatic cache invalidation on document updates.
- **Ingest Latency**: Uncached cold ingest required **7,877.2 ms** on CPU; warm re-ingest completed in **11.4 ms** (**692.7x speedup**).
- **Cache Hit Rate**: **94.85%** on repeat evaluation cycles.
- **Numerical Drift**: Exact 0.0000% difference; cached vectors match uncached vectors bit-for-bit.

### 6. Evidence-Span Learning Readiness Audit
- **Extractive Heuristic Baseline**: Concise span extraction compressed answers to a median of 17.0 words while maintaining 60/60 (100.0%) citation validity and 0.97 ms latency.
- **Learned Boundary Predictor Verdict**: **BLOCKED (Premature)**.
  - Architecture: `ChakrBiEncoder` uses mean pooling into a global sequence embedding; it does not output token-level logits required for span pointer prediction.
  - Data: The repository currently possesses only 16 labeled training examples in `split_manifest.json`. Training a learned span model without leaking the frozen 70 held-out questions requires a dedicated annotation set of 500+ examples.

### 7. Test Results
- `tests/test_batch1021_1060_local_assistant.py`: **5 / 5 passed**
- `tests/test_batch1101_1140_semantic_retrieval.py`: **5 / 5 passed**
- `tests/test_batch1141_1180_corpus_generalization.py`: **6 / 6 passed**
- **Combined Test Count**: **16 / 16 passed** (100% pass rate).

### 8. Checkpoint Integrity
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Initial SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Final SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match: True)
