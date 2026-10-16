# Experiment Protocol — Batches 1261–1300
## Evidence-Governed Refutation and Hybrid Neural Reranking

### 1. Mission Objectives
1. **Evidence-Grounded Refutation Policy**:
   Implement and validate an explicit, evidence-grounded decision policy distinguishing `ANSWER`, `REFUTE`, and `ABSTAIN`.
   A false premise question must be refuted (`REFUTE`) only when verified indexed reference evidence directly contradicts the query premise. Otherwise, it must honestly abstain (`ABSTAIN`).
2. **Hybrid Neural Reranking**:
   Evaluate whether the domain-adaptive `ChakrBiEncoder` (`experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt`) can improve ranking when applied as a second-stage reranker over a frozen top-$K$ BM25 candidate set, avoiding the cross-domain dense attractor collapse observed in unconstrained retrieval.
3. **Repository Invariants**:
   - Canonical training checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` must remain unmodified (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
   - Domain-adaptive checkpoint `experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt` must match SHA-256 (`46543f36e6b172b3584654728c5166365d9761bbe587b5cfd10338a01218e41a`).
   - `chakrview/runtime/session.py` and unrelated untracked files must not be overwritten or touched.
   - BM25 remains production default unless reranking demonstrates clear empirical superiority on held-out data.

### 2. Decision Contract & Evaluation Rules
- **ANSWER**: Supported factual query where relevant evidence is found and token overlap/confidence requirements are met.
- **REFUTE**: The query asserts an explicit factual claim (number, entity, or mechanism) that is directly contradicted by an indexed document fact, with exact source citation and evidence span provided.
- **ABSTAIN**: Neither sufficient evidence nor contradiction can be verified; honesty is prioritized over speculative answering.

### 3. Evaluated Systems for Phase 4
A. Existing BM25 baseline ($K=5$).
B. Existing deterministic hash hybrid baseline ($K=5$).
C. Existing domain-adaptive bi-encoder hybrid baseline ($K=5$).
D. BM25 top-$K$ candidate generation ($K=10, 20$) followed by domain-adaptive neural reranking.
E. Calibrated score fusion between BM25 and neural reranking score.
