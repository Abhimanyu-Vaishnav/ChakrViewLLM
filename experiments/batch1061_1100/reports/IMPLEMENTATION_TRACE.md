# Implementation Trace: Local Knowledge Assistant MVP (Batches 1021–1060)
## Complete Execution Path Analysis

### 1. Architectural Architecture & Flow Overview

```
User Query (CLI / Python API)
  │
  ▼
[1] Conversation History Anaphora Check
  │   - Does query contain referential pronouns ('that', 'it', 'the document', etc.)?
  │   - If yes: prepend previous user turn content -> resolved_query.
  ▼
[2] EvidenceGovernedAnsweringGate (Evidence Gate)
  │   - Check hard-coded refutation patterns (Celestial bodies, historical anachronisms, division by zero).
  │   - If matched: return GateDecision.REFUTE ("Contradiction detected: ...") immediately.
  ▼
[3] HybridRetriever Execution
  │   - Lexical Search: BM25KnowledgeIndex (k1=1.5, b=0.75) computes BM25 term scores over all indexed chunks.
  │   - Semantic Search: DeterministicHashEmbeddingProvider embeds query into 128-dim vector;
  │     InMemoryVectorIndex computes cosine similarity.
  │   - Fusion: Score = 0.8 * Norm(BM25) + 0.2 * Norm(Cosine). Top 5 candidates selected.
  ▼
[4] Content Token Overlap & Candidate Re-Ranking
  │   - Query tokens filtered against stopword set (including common verbs and auxiliaries).
  │   - Dual-token scoring:
  │     Score = len(curr_query_overlap) * 5.0 + len(resolved_query_overlap) * 1.5 + candidate.score * 0.5.
  │   - Best candidate chunk selected.
  ▼
[5] Grounding Threshold & Decision Logic
  │   - If no candidate, or overlap count < 1, or (content_tokens >= 3 and overlap < 2):
  │       Decision: ABSTAIN
  │       Answer: "I could not find sufficient verified evidence in the indexed local documents..."
  │   - Else:
  │       Decision: ANSWER
  │       Answer: Entire text of best candidate chunk + citation "[Source: <doc_id> (char <start>-<end>)]"
  ▼
[6] Session Memory Recording
  │   - User query and Assistant answer recorded in ConversationTurn list.
  │   - ConversationContextManager.compact_turns() trims older turns to enforce 400-word budget.
  ▼
AssistantResponse returned to caller / CLI
```

### 2. Component Taxonomy: Neural vs. Deterministic vs. Rule-Based

| Component | Classification | Implementation Source | Notes |
| :--- | :--- | :--- | :--- |
| **Document Chunking** | Deterministic | `DocumentIngestionEngine.ingest_file` | Fixed character window (800 chars, 100 overlap) snapping to newlines/periods. |
| **BM25 Lexical Index** | Deterministic | `BM25KnowledgeIndex` | Standard Okapi BM25 implementation. |
| **Dense Vector Index** | Deterministic (Hash) | `DeterministicHashEmbeddingProvider` | MD5 hash-modulo pseudo-random vector projection; not a learned neural embedding. |
| **Score Fusion** | Deterministic | `HybridRetriever.retrieve_candidates` | Convex combination: $0.8 \times \text{BM25} + 0.2 \times \text{Cosine}$. |
| **Anaphora Resolution** | Heuristic / Rule-based | `LocalKnowledgeAssistant.ask` | String matching for `that`, `it`, `previous`, etc., prepending prior turn query. |
| **Refutation Gate** | Heuristic / Rule-based | `EvidenceGovernedAnsweringGate` | Explicit regex matching celestial capitals, PM of India pre-1947, and division by zero. |
| **Answer Generation** | Deterministic Span Extraction | `LocalKnowledgeAssistant.ask` | Emits entire raw text of the top-ranked retrieved chunk with attached metadata citation. |
| **Core Neural Model** | **Inactive in Answer Path** | `ChakrMicro` | The canonical transformer checkpoint (`training_checkpoint_step_10000.pt`) was verified on disk but **is not executed or evaluated during standard Q&A answer generation**. |

### 3. Key Findings on System Behavior

1. **Answer Generation is Deterministic Retrieval-Passage Emission:**
   The assistant does not synthesize or summarize an answer using the autoregressive `ChakrMicro` core. It returns the raw chunk text verbatim from the indexed markdown file, appending the source citation.
2. **Deterministic Vector Provider:**
   The default `HybridRetriever` in `LocalKnowledgeAssistant` instantiates `DeterministicHashEmbeddingProvider` (hash-based projection), meaning semantic vector retrieval is deterministic token hashing rather than neural dense retrieval.
3. **High Benchmark Accuracy Root Cause:**
   The 50/50 benchmark score in Batches 1021–1060 succeeded because:
   - Evaluator used substring matching (`expected_substring in response.answer`).
   - Returning entire 800-character chunks virtually guaranteed that the target fact was contained inside the emitted paragraph if the document chunk was retrieved correctly.
