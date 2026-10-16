# Experiment Protocol — Batches 1141–1180
## Corpus-Scale Generalization, Semantic Retrieval Stress Test & Evidence-Span Learning Readiness

### 1. Protocol Objective
To evaluate the scaling limits, semantic robustness, embedding caching efficiency, follow-up conversation mechanics, and span learning readiness of ChakrView's local knowledge assistant across a 52-document corpus.

### 2. Frozen Datasets & Hashed Manifests
| Artifact | Path | SHA-256 Hash | Size / Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Corpus Files** | `experiments/batch1141_1180/data/corpus/` | `CORPUS_MANIFEST.json` | 52 documents (~116 chunks) | 6 operational domains with distractors and competing version specs |
| **Dev Split** | `experiments/batch1141_1180/data/dev_30_benchmark.json` | `ce1a4a8b03d59741b41ff44344eeb1f0a75f743b837ea61c7718a4a135a237ee` | 30 queries | Tuning, diagnostic, and threshold calibration split |
| **Held-Out Split** | `experiments/batch1141_1180/data/held_out_70_benchmark.json` | `7e6fcc06706fe686eb2cd8ba4891c5b0b1fd8b23372d5f197f935510df79fe12` | 70 queries | Untouched frozen test suite for final evaluation |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Fixed | Canonical 10,000-step checkpoint |
| **Bi-Encoder Checkpoint**| `experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt` | `d55d8272f2d93e5066a34c8a2bda0728c89c8913ffad48ec436a5faeb30d43e2` | Fixed | Indigenous 2-layer transformer bi-encoder |

### 3. Evaluated Retrieval & Answering Configurations
1. **BM25 Pure Lexical**: Standard Okapi BM25 index with $k_1=1.5, b=0.75$.
2. **Hybrid Deterministic Hash**: BM25 (0.8) + Deterministic 64-dim SHA-256 hash vector projections (0.2).
3. **Hybrid Learned Neural (`ChakrBiEncoder`)**: BM25 (0.8) + 64-dim continuous transformer embeddings (0.2).
4. **Span Extraction vs. Chunk Dumping**: Concise unit span extraction (sentences/bullets) vs. full 800-character chunk dumping.

### 4. Metrics Protocol
- **Retrieval Recall@1, Recall@3, Recall@5**: Evaluated against gold-evidence source documents.
- **Mean Reciprocal Rank (MRR)**: $1 / \text{rank}$ of the gold document in top-5 candidate chunks.
- **Decoupled Citation Validity vs Citation Support**:
  - Validity: Exact byte offset slice $[start\_char : end\_char]$ matches the physical file content on disk.
  - Support: Cited span explicitly contains the expected factual answer.
- **Token-Level F1**: Precision, Recall, and harmonic mean across normalized tokens.
- **Cold vs. Warm Latency**: Separate accounting for initial document embedding vs warm query inference.
