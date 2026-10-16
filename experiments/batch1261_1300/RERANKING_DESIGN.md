# Reranking Design & Candidate Generation Protocol — Batches 1261–1300
## Architecture, Frozen BM25 Candidate Depth, and Scoring Formulations

### 1. The Reranking Hypothesis
In Batches 1221–1260, we discovered that unconstrained pure dense retrieval over the entire corpus collapsed to **5.0% Recall@1** because generic tokens in out-of-domain chunks (e.g. `billing_subscription_tiers.md`) acted as high-similarity attractors ($>0.95$ cosine similarity).

**The Reranking Hypothesis**:
If we first generate a high-recall, lexical candidate pool of depth $K$ ($K=10$ or $K=20$) using BM25, we eliminate 95% of cross-domain false-positive attractors. Then, within this narrow, lexically relevant set, the domain-adaptive `ChakrBiEncoder` can rerank candidate passages using dense semantic understanding.

### 2. Candidate Generation & Depth Selection ($K$)
- **Corpus Size**: 52 documents, 116 chunks.
- **Candidate Pool Depths Evaluated**:
  - $K=5$ (Baseline depth)
  - $K=10$ (Standard reranking depth: 8.6% of corpus)
  - $K=20$ (Broad reranking depth: 17.2% of corpus)
- **Candidate Recall Ceiling**:
  A reranker cannot retrieve a document if it is absent from the initial BM25 candidate set.
  On the 60 answerable benchmark items:
  - BM25 Candidate Recall at $K=5$: **98.33% (59/60)**
  - BM25 Candidate Recall at $K=10$: **98.33% (59/60)**
  - BM25 Candidate Recall at $K=20$: **100.00% (60/60)**

### 3. Evaluated Reranking Systems
1. **System A — Pure BM25 Baseline ($K=5$)**:
   Standard BM25 lexical ranking without any neural component.
2. **System B — Deterministic Hash Hybrid Baseline ($K=5$)**:
   $S = S_{\text{bm25}} + 0.2 \cdot S_{\text{hash}}$.
3. **System C — Existing Domain-Adaptive Full-Corpus Search**:
   Full corpus vector retrieval fused with BM25.
4. **System D — Pure Neural Reranker over Frozen BM25 Top-$K$**:
   Take the top-$K$ candidates retrieved by BM25. Score each candidate chunk using:
   $$S_{\text{dense}} = \cos(v_Q, v_C) = \frac{v_Q \cdot v_C}{\|v_Q\|_2 \|v_C\|_2}$$
   Sort the top-$K$ candidates strictly by $S_{\text{dense}}$, using BM25 rank as deterministic tie-breaker.
5. **System E — Calibrated Score Fusion over Frozen BM25 Top-$K$**:
   Normalize BM25 scores via min-max scaling to $[0, 1]$:
   $$\hat{S}_{\text{bm25}} = \frac{S_{\text{bm25}} - \min(S)}{\max(S) - \min(S) + \epsilon}$$
   Dense cosine similarity is already bounded in $[-1, 1]$. Linearly fuse:
   $$S_{\text{fused}} = (1 - \alpha) \cdot \hat{S}_{\text{bm25}} + \alpha \cdot S_{\text{dense}}$$
   Evaluate across $\alpha \in \{0.1, 0.2, 0.5\}$.
