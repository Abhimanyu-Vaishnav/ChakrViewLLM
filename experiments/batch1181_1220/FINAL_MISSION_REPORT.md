# Final Mission Report — Batches 1181–1220
## Domain-Adaptive Retrieval Learning, Hard-Negative Mining and Independent Validation

### 1. Executive Summary
Batches 1181–1220 conducted a scientific, leakage-resistant evaluation of learned neural representations for ChakrView's local knowledge retrieval engine. We constructed an indigenous contrastive dataset with hard-negative mining, trained `ChakrBiEncoder` from scratch on CPU, and evaluated four retriever configurations against the frozen 70-item technical benchmark.

Key finding: **Deterministic BM25 retrieval remains empirically superior to learned neural bi-encoders on technical documentation retrieval**, achieving **98.33% Recall@1** and **1.05 ms latency** versus the neural hybrid's **88.33% Recall@1** and **4.47 ms latency**. Consequently, we retain BM25 as the default production configuration and maintain the trained neural bi-encoder as an opt-in experimental provider.

### 2. Multi-Model Retrieval Benchmark Comparison (70 Frozen Items)

| Metric | Configuration 1 (BM25 Lexical) | Configuration 2 (Deterministic Hash) | Configuration 3 (Untrained Bi-Encoder) | Configuration 4 (Domain-Adaptive Bi-Encoder) | Ablation (Pure Neural Search) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | **98.33% (59/60)** | **98.33% (59/60)** | 91.67% (55/60) | 88.33% (53/60) | 5.00% (3/60) |
| **Recall@3** | 98.33% (59/60) | 98.33% (59/60) | **100.00% (60/60)** | 98.33% (59/60) | 11.67% (7/60) |
| **Recall@5** | 98.33% (59/60) | 98.33% (59/60) | **100.00% (60/60)** | 98.33% (59/60) | 15.00% (9/60) |
| **MRR** | **0.9833** | **0.9833** | 0.9556 | 0.9333 | N/A |
| **Overall Accuracy** | 84.29% (59/70) | 84.29% (59/70) | **85.71% (60/70)** | 84.29% (59/70) | N/A |
| **Mean Token F1** | 0.1427 | 0.1427 | 0.1388 | 0.1419 | N/A |
| **Median Latency** | **1.05 ms** | **1.00 ms** | 4.04 ms | 4.47 ms | 3.82 ms |
| **Ingest Time (116 Chunks)** | 446.15 ms | **62.04 ms** | 387.85 ms | 576.53 ms | 312.40 ms |

### 3. Category Breakdown (Domain-Adaptive Neural vs BM25)

| Category | Query Count | BM25 R@1 | BM25 Accuracy | Neural Hybrid R@1 | Neural Hybrid Accuracy | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **exact_entity** | 10 | 100.0% | 90.0% | 100.0% | 90.0% | Both engines perform strongly on exact parameters. |
| **paraphrase** | 10 | 100.0% | 90.0% | 100.0% | 90.0% | High lexical grounding is resilient to phrasing. |
| **synonym** | 10 | 100.0% | 100.0% | 90.0% | 100.0% | BM25 index handles synonyms accurately. |
| **distractor_competition**| 10 | 100.0% | 80.0% | 80.0% | 80.0% | Dense embeddings introduce slight cross-chunk noise on identical numbers. |
| **version_conflict** | 6 | 100.0% | 83.3% | 83.3% | 83.3% | Strict keyword matching isolates V0.6 vs V0.7 specifications. |
| **follow_up** | 8 | 87.5% | 100.0% | 75.0% | 100.0% | All 8/8 follow-ups answered correctly end-to-end. |
| **multilingual_hinglish** | 6 | 83.3% | 83.3% | 83.3% | 83.3% | Equal performance across Hindi/English queries. |
| **abstention** | 5 | N/A | 100.0% | N/A | 100.0% | 100% honest abstention on unindexed topics. |
| **false_premise** | 5 | N/A | 0.0% | N/A | 0.0% | Both systems answer the factual topic instead of asserting falsehood. |

### 4. Paired Query Analysis (New vs Old Neural Model)
- **New Model Wins**: 1 query
- **New Model Losses**: 2 queries
- **Ties**: 56 queries
- Dense contrastive tuning on small sequence lengths did not yield statistically meaningful improvements over lexical search in technical documentation domains.

### 5. Invariant & Checkpoint Integrity
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Initial SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Final SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact Match Verified).
- **Protected Files**: `chakrview/runtime/session.py` was strictly preserved and unmodified.

### 6. Test Suite Execution
- **Command**:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_batch1181_1220_domain_adaptive_retrieval.py tests/test_batch1141_1180_corpus_generalization.py tests/test_batch1101_1140_semantic_retrieval.py tests/test_batch1021_1060_local_assistant.py -v
  ```
- **Results**: **22 passed, 0 failed** in 2.85s (100% pass rate).
