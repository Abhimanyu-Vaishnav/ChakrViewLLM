# Batch 153 — Retrieval Evaluation & Lifecycle Report
## Cognitive Routing, Hybrid Retrieval & Provenance Integrity

### 1. Architectural Foundation
ChakrView's knowledge tier decouples mutable external information from immutable neural core weights. In Batch 153, we validated the `RetrievalAdapter` wrapping `HybridRetriever`, combining:
1. **Okapi BM25 Lexical Retrieval**: Length-normalized term frequency matching with Unicode/Devanagari term tokenization and stopword suppression.
2. **Deterministic Vector Retrieval**: Vector space matching utilizing the reference embedding provider with cosine similarity.
3. **Score Normalization & Fusion**: Hybrid score combination $S_{\text{hybrid}} = w_{\text{lex}} \cdot \text{Norm}(S_{\text{lex}}) + w_{\text{sem}} \cdot \text{Norm}(S_{\text{sem}})$.
4. **Passive Provenance Containment**: SHA-256 chunk hashing and chunk metadata preservation.

### 2. Retrieval Evaluation Benchmark
The evaluation corpus contains structured scientific, geographic, and multilingual documents along with distractor texts:

| Query | Target Document | Expected Keyword/Fact | Hybrid Score | Provenance Hash (Prefix) | Retrieval Outcome |
|---|---|---|---|---|---|
| "What is the atomic number of Carbon?" | `doc_carbon` | `atomic number of Carbon is 6` | 0.6558 | `784d1fa6...` | **PASS (Rank 1)** |
| "In which state is the Thar Desert located?" | `doc_thar` | `Rajasthan, India` | 0.6720 | `e5bc118a...` | **PASS (Rank 1)** |
| "What was the name of the crater where the spacecraft landed?" | `doc_shackleton` | `crater named Shackleton` | 0.7028 | `bd2ff399...` | **PASS (Rank 1)** |
| "धातु क्या है?" (Hindi metal query) | `doc_hindi_metal` | `धातु: तांबा` | 0.6224 | `3e6fa16b...` | **PASS (Rank 1)** |
| "Who invented the telephone?" (Absent Query) | *None* | *None* (Out-of-domain) | < 0.20 | N/A | **PASS (Clean Abstention)** |

### 3. Key Findings & Weaknesses Fixed
1. **Stopword Leakage in BM25**: Early iterations matched unrelated queries on high-frequency English function words ("the", "is", "of"). We implemented stopword filtering in `BM25KnowledgeIndex._tokenize` to prevent spurious BM25 matches.
2. **Reference Vector Hash Collisions**: Without semantic training, hash embeddings produce background similarity noise ($0.05 - 0.15$). We enforced that reference hash embeddings require non-zero lexical support (`raw_lexical_score > 0.0`) to prevent false-positive chunk matches for completely out-of-domain queries.
3. **Multilingual Unicode Indexing**: BM25 regex was updated to `\w+` with `re.UNICODE` flags, correctly indexing Hindi/Indic and scientific symbols.

### 4. Conclusion & Guarantees
- **Recall@1 on known factual queries**: 100% (4/4).
- **Abstention on absent queries**: 100% (1/1).
- **Chunk Provenance**: 100% verifiable SHA-256 hashes attached to every retrieved piece of evidence.
