# Evidence-Based Next Milestone Decision — Batches 1181–1220
## Production Deployment Verdict & Roadmap for Batches 1221–1260

### 1. Executive Summary & Verdict
**VERDICT: RETAIN BM25 AS THE DEFAULT RETRIEVAL ENGINE; KEEP DOMAIN-ADAPTIVE BI-ENCODER AS OPT-IN EXPERIMENTAL COMPONENT.**

### 2. Empirical Rationale
In Batches 1181–1220, we trained `domain_adaptive_bi_encoder.pt` using 444 contrastive triples with hard negatives mined from identical topic families.
While the model achieved 100% classification accuracy on pairwise triples, comprehensive multi-model offline evaluation on the frozen 70-item technical benchmark revealed:
1. **Retrieval Accuracy**:
   - BM25 achieved **98.33% Recall@1 (59/60)** and **0.9833 MRR**.
   - Deterministic Hash Hybrid achieved **98.33% Recall@1 (59/60)** and **0.9833 MRR**.
   - Domain-Adaptive Bi-Encoder Hybrid achieved **88.33% Recall@1 (53/60)** and **0.9333 MRR**.
2. **Pure Dense Retrieval Ablation**:
   - Pure vector search without lexical keywords achieved only **5.00% Recall@1 (3/60)**. The small 64-dimensional sequence embeddings suffer representation collapse when distinguishing dense jargon across 116 technical chunks.
3. **Latency Impact**:
   - BM25 median query latency: **1.05 ms**.
   - Neural hybrid median query latency: **4.47 ms** ($4.25\times$ slower on CPU).

In accordance with Acceptance Gate B and Gate J:
- **Neural superiority is NOT claimed**.
- BM25 remains superior for exact entity, numeric identifier, and version-specific technical document retrieval.
- We will not promote the learned neural bi-encoder to default production retriever.

### 3. Decisions & Recommendations for Batches 1221–1260
1. **Default Retrieval Architecture**: Keep BM25 lexical retrieval (or BM25 + deterministic hash hybrid) as the primary runtime engine in `LocalKnowledgeAssistant`.
2. **Dense Representation Scaling**: Before re-attempting dense retrieval promotion, the encoder must be scaled from a toy 2-layer 64-dim network to a token-aware sequence representation with at least $d_{\text{model}}=256, d_{\text{embed}}=128$, trained with in-batch negatives over larger corpus chunks ($>10,000$ tokens).
3. **Extractive Answer-Span Boundary Prediction**: Maintain the deterministic sentence/bullet boundary extractor. Do not introduce speculative neural span models until token-level sequence representations are implemented.
