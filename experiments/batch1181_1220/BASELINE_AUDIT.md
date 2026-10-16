# Baseline Audit — Batches 1181–1220
## Verification of Batch 1141–1180 Benchmark and Ground Truth Contract

### 1. Invariant Integrity and Canonical Checkpoint
- **Canonical Checkpoint Path**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Measured SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Exact Match Verified)
- **Protected Files**: `chakrview/runtime/session.py` remains uncommitted/unmodified and strictly untouched.

### 2. Frozen Evaluation Benchmark Audit
- **File**: `experiments/batch1141_1180/data/held_out_70_benchmark.json`
- **SHA-256**: `7e6fcc06706fe686eb2cd8ba4891c5b0b1fd8b23372d5f197f935510df79fe12`
- **Total Test Items**: 70 (Unique IDs: 70, from 101 to 170).
- **Subsets & Denominators**:
  - `expected_type == 'ANSWER'`: 60 items. (Evaluated for Recall@1, Recall@3, Recall@5, MRR, Answer F1, Citation Validity, Citation Support).
  - `expected_type == 'ABSTAIN'`: 5 items. (Out-of-domain / unindexed topics).
  - `expected_type == 'REFUTE'`: 5 items. (False premises contradicting established corpus facts).
  - Total queries evaluated for Overall End-to-End Task Accuracy: 70 items.
- **Category Breakdown**:
  - `exact_entity`: 10 items
  - `paraphrase`: 10 items
  - `synonym`: 10 items
  - `distractor_competition`: 10 items
  - `version_conflict`: 6 items
  - `follow_up`: 8 items
  - `multilingual_hinglish`: 6 items
  - `false_premise`: 5 items
  - `abstention`: 5 items
  - **Sum of Categories**: Exactly 70 items.

### 3. Independence of Metrics
- **Retrieval Rank**: Measured strictly by whether the expected source document chunk appears at rank 1, $\le 3$, or $\le 5$ in candidate pool.
- **Answer Quality (Token F1)**: Computed strictly against `expected_substring`.
- **Citation Validity**: Evaluated against actual file path existence and byte offset slicing on disk.
- **Abstention**: Evaluated explicitly via `status == 'HONEST_ABSTENTION'`.

### 4. Corpus Corpus & Chunking Audit
- **Corpus Directory**: `experiments/batch1141_1180/data/corpus/`
- **Total Documents**: 52 markdown files.
- **Total Chunks Generated**: 116 chunks ($L \approx 400$ chars, overlap = 100).
- **Corpus Manifest**: `experiments/batch1141_1180/data/CORPUS_MANIFEST.json`
- **Corpus Manifest SHA-256**: `bf9fc8a7ce5cb4df802263bbfeebc9665f8a0ffbe0145dc656b82531a7f05167`
- All retriever baselines (BM25, Hash Hybrid, and BiEncoder Hybrid) consume the exact identical chunks, queries, and evaluation harness.

### 5. Identified Baseline Limitations & Objective
The existing `improved_bi_encoder.pt` was trained on only 8 factual science examples from physics and astronomy. It has never seen systems, networking, security, storage, or enterprise docs. As a result, its dense embeddings lack domain-specific term discriminability, yielding 91.67% Recall@1 compared to BM25's 93.33%.
The objective of Batches 1181–1220 is to construct a domain-adaptive training dataset with hard negatives and determine if contrastive fine-tuning can improve or match lexical Recall@1 while retaining its 100% Recall@3 coverage.
