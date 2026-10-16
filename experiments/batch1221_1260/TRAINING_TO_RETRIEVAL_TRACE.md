# Training-to-Retrieval Gap Trace — Batches 1221–1260
## End-to-End Mechanical Diagnosis: Why 100% Pairwise Accuracy Collapses to 5% Dense Retrieval

### 1. The Paradox
In Batches 1181–1220, `domain_adaptive_bi_encoder.pt` reported:
- **Pairwise Contrastive Test Split Accuracy: 100.0%** (68/68 triples, mean margin = 0.8605).
- **Corpus-Scale Pure Dense Retrieval Recall@1: 5.0%** (3/60 answerable benchmark questions).
- **Hybrid Retrieval Recall@1: 88.33%** (53/60), underperforming pure BM25 (98.33%, 59/60).

### 2. End-to-End Pipeline Step Analysis

| Stage | Training Pipeline | Retrieval Pipeline | Discrepancy / Gap Identified |
| :--- | :--- | :--- | :--- |
| **1. Text Units** | Short positive sentence ($\approx 15$ words) vs single negative sentence. | Full corpus chunks ($\approx 400$ chars, 60–80 words) containing multiple bullet points and headers. | **Unit Mismatch**: The model was never trained to match queries against multi-sentence paragraph chunks. |
| **2. Candidate Pool** | Exactly **1 negative** per query ($K=2$ classification). | **72 to 116 corpus chunks** across 52 documents ($K=116$ ranking competition). | **Competition Scale**: Binary margin $> 0.2$ against 1 distractor is mathematically trivial compared to being the top-1 among 116 dense competitors. |
| **3. Negative Sampling** | Hard negatives chosen randomly from another doc in the same topic family. | In-batch negatives were NOT used; corpus contains hundreds of irrelevant documents from other domains. | **Unconstrained False Positives**: The model was never penalized for assigning high similarity to documents in *other* domains (e.g. billing docs when asking about compute memory). |
| **4. Sequence Representation** | Mean pooling over token embeddings followed by projection to $d_{\text{embed}}=64$. | Same mean pooling. | **Vector Dimension Bottleneck**: 64 continuous float dimensions cannot represent the unique topological distinctions of 52 diverse technical documents. |
| **5. Inter-Chunk Similarity** | Positive sim: 0.941, Negative sim: 0.080. | Mean inter-chunk background cosine: **0.635** ($\pm 0.352$). | **Embedding Space Distortion**: Irrelevant documents (e.g. `billing_subscription_tiers.md`) have an average cosine similarity of $>0.90$ with technical questions because common tokens like "per", "with", "system" dominate the mean pooling vector. |

### 3. Mechanistic Root Cause Summary
1. **False-Positive Attractor Chunks**:
   Because the bi-encoder was trained solely on same-topic hard negatives (e.g., storage vs storage), it never learned that `billing_subscription_tiers.md` should have zero similarity to `compute_arch_core.md`. In unconstrained corpus retrieval, generic tokens in billing documents pull query vectors toward billing chunks with similarity $>0.95$, completely bypassing the true technical chunk (similarity $\approx 0.79$).
2. **Dense Hybrid Noise**:
   In hybrid retrieval, adding this uncalibrated dense score ($8.0 \times \text{dense}$) to BM25 scores introduces high-variance noise that knocks valid top-1 BM25 candidates down to rank 2 or 3.
