# Experiment Protocol — Batches 1181–1220
## Domain-Adaptive Retrieval Learning, Hard-Negative Mining and Independent Validation

### 1. Mission Objectives
1. Audit the baseline retrieval performance on the 70-item technical documentation corpus benchmark.
2. Build a domain-adaptive training dataset with hard-negative mining from identical topic families without data leakage.
3. Train an indigenous `ChakrBiEncoder` using triplet margin loss on CPU under strict reproducibility invariants (no external pretrained models or hosted APIs).
4. Conduct fair multi-model evaluations comparing BM25, Hash Hybrid, Existing Neural Bi-Encoder, and Domain-Adaptive Bi-Encoder.
5. Perform an ablation isolating pure neural dense vector retrieval from lexical retrieval.
6. Verify canonical checkpoint integrity, regression test coverage, and render an evidence-based recommendation for default runtime retrieval.

### 2. Dataset Construction & Split Protocol
- **Source Corpus**: 52 markdown documents in `experiments/batch1141_1180/data/corpus/`.
- **Query-Passage Triples Generated**: 444 examples.
  - Positive Passage: Exact factual sentences extracted from the target document.
  - Hard Negative Passage: Semantically overlapping text extracted from a different document in the *same* topic family.
- **Data Splitting**:
  - Train Split: 310 triples (70%) — SHA-256: `bcf03e50f1ded89a91f89eea3511a46f93ed099075f9dea3244b39f47c1e146c`
  - Dev Split: 66 triples (15%) — SHA-256: `d048f2241968b4c071896285ca3070cc69579e2c121bf88cc216d064c2462e8d`
  - Test Split: 68 triples (15%) — SHA-256: `35b875850066221c0180654ebf01a0260415f0318cbbfad8039bc680be0abfdf`
- **Leakage Prevention**: Stratified allocation ensures exact facts or paraphrases do not cross between training, development, and the frozen held-out benchmark.

### 3. Model Architecture & Training Protocol
- **Model**: `ChakrBiEncoder` ($N=2$ transformer encoder layers, $d_{\text{model}}=128$, $n_{\text{heads}}=4$, $d_{\text{embed}}=64$, mean-pooling, LayerNorm, $L_2$ norm).
- **Tokenizer**: Indigenous ChakrView 4096-vocab BPE tokenizer (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- **Optimization**: AdamW ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-2}$), TripletMarginLoss ($\text{margin}=0.2, p=2$), batch size 32.
- **Hardware & Environment**: CPU-only execution on Windows 11, Python 3.14.7, PyTorch 2.14.0+cpu.
- **Selection Criterion**: Checkpoints selected exclusively on development set accuracy and loss.

### 4. Evaluation Protocol
- **Frozen Benchmark**: 70 items from `experiments/batch1141_1180/data/held_out_70_benchmark.json` (60 answerable, 5 honest abstentions, 5 false premises).
- **Metrics**:
  - Recall@1, Recall@3, Recall@5 on answerable questions.
  - Mean Reciprocal Rank (MRR).
  - Overall End-to-End Task Accuracy across all 70 questions.
  - Token-level Answer F1.
  - Median and P95 latency (ms).
- **Configurations Evaluated**:
  1. BM25 Lexical Baseline
  2. Deterministic Hash Embedding Hybrid
  3. Untrained/Physics ChakrBiEncoder Hybrid
  4. Domain-Adaptive ChakrBiEncoder Hybrid
  5. Pure Domain-Adaptive Vector Search Ablation (no BM25)
