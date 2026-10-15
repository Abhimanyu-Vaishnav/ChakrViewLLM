# Next Milestone Decision: Batches 1101–1140
## Strategic Roadmap & Technical Prioritization

### 1. Direct Answers to Decision Criteria

**Q1: What was actually reproduced from the previous report?**
- The full 50/50 test benchmark result, the 38/38 citation bounds, and the sub-millisecond query latency reproduced exactly under an independent harness.

**Q2: What is the true generalization capability of the assistant?**
- On a completely held-out 60-question suite with unseen paraphrases, synonyms, and multi-turn references, the assistant achieved **59/60 (98.33%)** task success with **100% honest abstention** on unindexed questions and **97.7% citation support**.

**Q3: Which behaviors are neural, hybrid, or deterministic?**
- **Neural:** None in the primary Q&A path. The core model (`ChakrMicro`) was not active during inference.
- **Hybrid:** Score fusion in `HybridRetriever` (combining BM25 lexical term scores with a deterministic pseudo-random hash embedding projection) and `EvidenceGovernedAnsweringGate`.
- **Deterministic:** Document parsing/chunking, Okapi BM25 term frequency counting, token overlap thresholds, and verbatim chunk text emission.

**Q4: What remains unverified or fragile?**
- **Semantic Distance Handling:** High-semantic-distance paraphrases (e.g. *"network traffic"* for *"in-transit communication"*) fail because the hash-based vector index lacks true learned semantic representations.
- **Answer Precision:** The assistant emits full 800-character chunks rather than extracted concise answers or synthesized sentences.
- **Ingestion Scaling:** Ingestion scales linearly and rebuilds indices after each file.

**Q5: Which next milestone is justified by the measured bottleneck?**
- **Recommended Next Milestone (Batches 1101–1140):**
  **Learned Dense Semantic Retrieval & Neural Extractive Span Compression.**
  1. **Activate / Fine-Tune Neural Semantic Embeddings:** Replace `DeterministicHashEmbeddingProvider` with a trained lightweight embedding representation using the existing custom tokenizer and `ChakrMicro` transformer layers.
  2. **Neural Answer Span Compression:** Instead of returning raw 800-character Markdown chunks, use `ChakrMicro`'s span grounding head (developed in Batches 861–900) to extract precise 1–2 sentence factual answer spans while retaining chunk provenance.
  3. **Batch Ingestion & Index Optimization:** Batch chunk ingestion to eliminate redundant index rebuilds during large folder ingestion.
