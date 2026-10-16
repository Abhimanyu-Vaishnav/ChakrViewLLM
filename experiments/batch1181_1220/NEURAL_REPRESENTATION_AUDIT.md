# Neural Representation Audit — Batches 1181–1220
## Representation Analysis, Empirical Contribution & Failure Modes

### 1. Training Objective & Provenance
- **Model Architecture**: `ChakrBiEncoder` (Transformer encoder with $N=2$ layers, $d_{\text{model}}=128$, $n_{\text{heads}}=4$, $d_{\text{embed}}=64$, mean-pooling over sequence dimension followed by LayerNorm and $L_2$ normalization).
- **Tokenizer**: Indigenous ChakrView BPE tokenizer with 4,096 vocabulary items (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- **Pretrained External Weights / APIs**: Zero. Evaluated completely from scratch.
- **Dataset Constructed**: 444 contrastive triples with hard negatives mined from identical topic families (`experiments/batch1181_1220/data/`).
  - Train: 310 triples
  - Dev: 66 triples
  - Test: 68 triples
- **Training Loss**: `TripletMarginLoss(margin=0.2, p=2)` executed for 60 epochs on CPU in 264.91 seconds.
  - Final Train Loss: 0.0000
  - Best Dev Accuracy: 1.0000 (mean margin = 0.8061)
  - Held-out Test Accuracy: 1.0000 (mean margin = 0.8611, positive similarity = 0.9410, negative similarity = 0.0799).

### 2. Generalization Reality: Held-Out Multi-Model Benchmark (70 Items)
While the domain-adaptive model achieved 100% classification on paired triples, evaluating it inside the end-to-end retrieval and local knowledge assistant on the frozen 70-item benchmark revealed critical findings:

| Model Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | Overall Accuracy | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Lexical Baseline** | **98.33%** (59/60) | 98.33% (59/60) | 98.33% (59/60) | **0.9833** | 84.29% (59/70) | **1.05 ms** |
| **Deterministic Hash Hybrid** | **98.33%** (59/60) | 98.33% (59/60) | 98.33% (59/60) | **0.9833** | 84.29% (59/70) | **1.00 ms** |
| **Untrained/Physics ChakrBiEncoder** | 91.67% (55/60) | **100.0%** (60/60) | **100.0%** (60/60) | 0.9556 | **85.71%** (60/70) | 4.04 ms |
| **Domain-Adaptive ChakrBiEncoder** | 88.33% (53/60) | 98.33% (59/60) | 98.33% (59/60) | 0.9333 | 84.29% (59/70) | 4.47 ms |
| **Ablation: Pure Neural Vector Search (No BM25)**| 5.00% (3/60) | 11.67% (7/60) | 15.00% (9/60) | — | — | 3.82 ms |

### 3. Detailed Ablation & Mechanics: Why Pure Dense Retrieval Fails
1. **The Pure Neural Vector Search Defect (5.0% Recall@1)**:
   When BM25 lexical indexing is removed and retrieval relies solely on inner products of 64-dimensional dense vectors, retrieval drops to 5.0% Recall@1.
   Because `d_embed=64` is trained on short sentence triples with mean pooling, the 64-dimensional space undergoes severe representation collapse across 116 technical chunks with overlapping infrastructure jargon ("cluster", "node", "storage", "bytes").
2. **Dense Noise in Hybrid Ranking**:
   When hybrid retrieval fuses BM25 scores with cosine similarities from the fine-tuned bi-encoder, the dense similarity scores are compressed into a narrow range ($[0.85, 0.95]$) for all documents within the same domain. This dense noise slightly degrades BM25's sharp discrimination on exact numeric identifiers and version tags (e.g. `distractor_competition` R@1 fell from 100% to 80%, and `synonym` R@1 fell from 100% to 90%).
3. **Lexical Grounding Remains Invaluable**:
   BM25 remains the fastest ($1.05\text{ ms}$) and most accurate ($98.33\%\text{ R@1}$) retrieval mechanism for technical systems documentation.

### 4. Conclusion on Neural Superiority
The experimental evidence definitively proves that **neural bi-encoder retrieval does NOT outperform BM25 on technical corpus retrieval**. 
In accordance with Gate B and Gate J, we will NOT claim neural superiority and will maintain BM25 as the default production configuration.
