# Failure Analysis & Robustness Audit — Batches 1261–1300
## Detailed Failure Modes in Neural Reranking and Decision Policy

### 1. Neural Reranking Failure Analysis
Our experiments evaluated whether neural reranking over a frozen BM25 candidate set could rescue dense retrieval from the cross-domain attractor collapse observed in Batches 1181–1220.

#### Candidate Set Recall Ceiling
- BM25 candidate recall at $K=5$: **98.33% (59/60)**
- BM25 candidate recall at $K=10$: **100.00% (60/60)**
- BM25 candidate recall at $K=20$: **100.00% (60/60)**

Within the top-10 candidate pool, 100% of the true gold passages were present.

#### Empirical Failure of Pure Neural Reranker ($K=10, 20$)
When sorting the top-10 BM25 candidates strictly by the domain-adaptive `ChakrBiEncoder` cosine similarity:
- **Recall@1 dropped catastrophically from 98.33% to 15.00% (9/60)**.
- **MRR dropped from 0.9833 to 0.3724**.
- **Latency increased from 1.04 ms to 62.44 ms** ($60\times$ slower on CPU due to $K$ forward passes per query).

#### Why Reranking Degrades Within BM25 Candidates
Even when candidates are constrained to those with lexical overlap, the 64-dimensional mean-pooled embeddings are incapable of fine-grained ranking:
1. **Query 102**: `"What is the minimum physical RAM required for CPU inference?"`
   - BM25 Candidate Pool (Top 5):
     1. `compute_memory_bus.md` (BM25: 18.2) — [Gold Document]
     2. `compute_cpu_simd.md` (BM25: 12.4)
     3. `billing_refund_policy.md` (BM25: 8.1)
   - Neural Cosine Similarity Reranking:
     1. `billing_refund_policy.md`: **0.9408**
     2. `compute_cpu_simd.md`: **0.8214**
     3. `compute_memory_bus.md`: **0.7014**
   - **Result**: Even inside the top-10 BM25 set, the generic phrasing in `billing_refund_policy.md` received higher cosine similarity than the specific memory bus document, demoting the true gold passage from Rank 1 to Rank 3!

2. **Query 103**: `"What minimum compute capability is required for CUDA acceleration?"`
   - BM25 ranked `compute_accelerator_cuda.md` at #1.
   - Neural reranker promoted `billing_metered_usage.md` to #1 (cosine 0.941 vs gold 0.899). Demoted gold passage to Rank 10!

#### Score Fusion Behavior ($\alpha \in \{0.2, 0.5\}$)
- Calibrated fusion with small neural weight ($\alpha=0.2$) preserved some BM25 structure, achieving **91.67% Recall@1 (55/60)**, but this still underperformed unaugmented BM25 (98.33%) while imposing a 60 ms CPU latency penalty.

---

### 2. Decision Policy Failure Case: Item 164
- **Query**: `"Why are database tokens configured to expire after 100 years?"`
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ABSTAIN`
- **Analysis**:
  The corpus contains `security_auth_tokens.md`, which defines "API bearer tokens" (60 minutes). It contains no mention of "database tokens".
  Because the lexical overlap between "database tokens" and "API bearer tokens" was below the confidence threshold, the assistant honestly responded:
  *"I could not find sufficient verified evidence in the indexed local documents to answer this question."*
- **Integrity Assessment**:
  This is a **safe failure mode**. Refuting a claim without finding verified matching evidence violates Decision Contract Rule 4 ("Missing evidence is not proof that a claim is false"). Abstaining honestly when terms do not match preserves enterprise safety.
