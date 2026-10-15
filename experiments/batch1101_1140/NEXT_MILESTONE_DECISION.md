# Next Milestone Decision — Batches 1101–1140
## Recommendation for Batches 1141–1180

### 1. Empirical Verdict from Batches 1101–1140
1. **Extractive Answer Span Precision**: Implemented and verified. Moving from 800-character chunk dumping to targeted sentence/bullet units reduced answer length by 3.6x (from 115 to 31 words) while maintaining 100% citation validity and target fact retention (43/43). This is ready for default production usage in the local knowledge assistant.
2. **Strict Evaluation Protocol**: The vulnerability where single-digit numbers (`"6"`) passed merely because they appeared inside an 800-character chunk has been closed. Token F1 and explicit numeric comparisons now protect the evaluation suite.
3. **Retrieval Reality**: On the 5-document curated corpus, BM25 lexical retrieval already reaches 95.45% Recall@1. While the indigenous `ChakrBiEncoder` generates genuine continuous embeddings and achieves identical top-1 recall, dense retrieval adds ~4 ms latency per query. Dense semantic retrieval will prove its distinctive value only when scaling to larger, uncurated document sets with high synonym variance and vocabulary mismatch.

### 2. Next Milestone Recommendations (Batches 1141–1180)

**Priority 1: Corpus Scaling & Paraphrase Stress-Testing**
- Ingest a corpus 10x larger (50+ documents, 500+ chunks) containing distracting passages, cross-domain terminology, and ambiguous vocabulary.
- Measure the divergence point where BM25 begins failing on pure semantic paraphrases and dense vector representations provide a decisive advantage.

**Priority 2: Neural Extractive Span Boundary Prediction**
- Train a lightweight pointer or boundary classifier on top of `ChakrBiEncoder` embeddings to predict start/end character offsets directly from dense representations, transitioning span extraction from heuristic parsing to learned boundary prediction.

**Priority 3: Multi-Document Synthesis & Contradiction Resolution**
- Extend the answering pipeline to synthesize answers spanning multiple disjoint documents and explicitly contrast conflicting evidence across documents when versions diverge.
