# Next Milestone Decision — Batches 1141–1180
## Recommendations for Batches 1181–1220

### 1. Empirical Verdict from Batches 1141–1180
1. **Lexical vs. Dense Retrieval Scaling**: On a 52-document technical corpus, BM25 remains exceptionally fast (0.98 ms) and highly accurate (93.33% Recall@1, 95.83% MRR). While the learned neural bi-encoder reaches 100% Recall@3/Recall@5, it does not beat BM25 on Recall@1 and adds 2–3 ms latency per query.
2. **Embedding Cache Impact**: LRU caching reduced corpus re-ingest time from 7.8 seconds to 11.4 milliseconds (692x speedup) with zero numerical drift.
3. **Follow-Up Anaphora**: Resolved the multi-turn pronoun failure via word-boundary trigger matching and stopword refinement, achieving 8/8 (100%) accuracy on held-out conversational questions.
4. **Span Boundary Learning**: Proven premature. Data scarcity (16 total examples) and pooled bi-encoder representations mean heuristic extraction remains strictly superior.

### 2. Next Milestone Strategic Directions (Batches 1181–1220)

**Priority 1: Contradiction Detection & False-Premise Refutation Engine**
- Currently, false-premise queries scored 0% because the assistant retrieved the correct passage and answered the underlying topic instead of detecting the invalid assumption in the prompt.
- Develop a dedicated premise-verification module in `EvidenceGovernedAnsweringGate` that checks subject-attribute claims against retrieved facts before generating answers.

**Priority 2: Multi-Document Answer Synthesis**
- Expand answer extraction to synthesize answers spanning multiple disparate documents (e.g. comparing V0.6 architecture with V0.7 draft deprecations).

**Priority 3: Domain-Specific Contrastive Fine-Tuning**
- Train the bi-encoder on domain-specific query-document triplets rather than physical science facts, investigating whether domain contrastive loss can boost Recall@1 over BM25 on technical text.
