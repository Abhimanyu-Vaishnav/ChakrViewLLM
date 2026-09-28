# STEP 13 — HYBRID MEMORY & SEMANTIC RETRIEVAL FOUNDATION

**ChakrView Indigenous Cognitive Engine**  
**Engineering & Scientific Specification Report**

---

## 1. MOTIVATION & OBJECTIVES

Prior to Step 13, the ChakrView runtime possessed two independent stateful components:
1. **Lexical Document Retrieval (Step 11 RAG)**: Powered by an Okapi BM25 index over chunked documents, highly effective at exact term matching and lexical keyword queries.
2. **Conversational Working Memory (Step 12)**: Managed session-isolated working memories (facts, constraints, preferences, summaries) ranked via heuristic recency, importance, and token-overlap scoring.

While functional, this decoupled retrieval structure suffered from three foundational limitations:
- **Lexical Vulnerability to Vocabulary Mismatch**: Pure BM25 relies on exact token overlap and fails when users query concepts via synonyms, paraphrases, or alternative terminology not present in the indexed document text.
- **Siloed Candidate Representation**: Working memories and knowledge chunks existed in disparate data representations, preventing unified relevance ranking and coherent context budget competition.
- **Absence of Embedding Abstractions**: No clean architectural boundary existed for dense vector representation, cosine similarity indexing, or score fusion.

**Step 13 Objective**: Establish a unified, modular hybrid retrieval architecture capable of:
1. Combining lexical BM25 retrieval and dense vector similarity through deterministic score fusion.
2. Unifying working memory and external knowledge document retrieval without merging their underlying storage or governance models.
3. Providing an abstract `EmbeddingProvider` contract with a lightweight, deterministic reference implementation for reproducible testing.
4. Preserving the strict 512-token context horizon, passive data containment boundaries, and zero heavyweight external dependencies.

---

## 2. EXISTING STEP 11 & STEP 12 ARCHITECTURE REVIEW

In Steps 11 and 12, the cognitive inference cycle followed this flow:
```text
User Query
    ↓
Skill Resolution (RuleBasedSkillResolver)
    ↓
Working Memory Extraction (Rule-based MemoryExtractor)
    ↓
Governed Tool Execution (Calculator, Text Tools)
    ↓
Working Memory Ranking (WorkingMemory.retrieve_relevant)
    ↓
Knowledge Retrieval (BM25KnowledgeIndex.search)
    ↓
PromptContextBuilder (Strict 512 token ceiling: System + Memory + Knowledge + History + Query)
    ↓
InferenceSession (ChakrMicro KV-cache incremental prefill & decode)
    ↓
RAGResponse / ChatResponse with Fine-Grained Provenance
```

Step 13 builds directly on this foundation:
- The frozen neural core `ChakrMicro v0.1` (3,443,136 parameters, 6 layers, $d_{\text{model}}=192$, context 512) remains completely untouched.
- The frozen Byte-Level BPE tokenizer ($V=4096$) remains unchanged.
- The runtime retains full ownership of state; the neural model remains strictly stateless.
- Existing BM25 knowledge search and conversational working memory continue to function with 100% backward compatibility.

---

## 3. NEW RETRIEVAL ABSTRACTIONS

Step 13 introduces a clean, unified retrieval abstraction layer in `chakrview/runtime/retrieval.py`:

```
                    ┌─────────────────────────┐
                    │     RetrievalQuery      │
                    │ (text, weights, top_k)  │
                    └────────────┬────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   ┌───────────────────┐                     ┌───────────────────┐
   │ Knowledge Sources │                     │  Memory Sources   │
   │ (BM25 + Vectors)  │                     │ (Working Memory)  │
   └────────┬──────────┘                     └─────────┬─────────┘
            │                                          │
            └────────────────────┬─────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    UnifiedRetriever     │
                    │  (Score Normalization & │
                    │   Deterministic Sort)   │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │     RetrievalResult     │
                    │  [RetrievalCandidate]   │
                    └─────────────────────────┘
```

### Core Abstractions:
- **`RetrievalSourceType`**: Enumeration designating the candidate's origin:
  - `KNOWLEDGE`: Verified external documentation chunks.
  - `DOCUMENT`: Whole-document references.
  - `MEMORY`: Conversational working memory items (facts, preferences, constraints).
  - `CONVERSATION_SUMMARY`: Condensed rolling dialogue summaries.
- **`RetrievalCandidate`**: Universal data container for any retrieved entity:
  - `candidate_id`: Unique identifier (e.g. `doc_001_c00` or `mem_042`).
  - `text`: Content payload.
  - `source_type`: Origin `RetrievalSourceType`.
  - `source_id`: Parent document ID or conversation turn ID.
  - `score`: Unified normalized relevance score $\in [0.0, 1.0]$.
  - `retrieval_method`: Scoring mechanism applied (`"hybrid"`, `"bm25"`, `"semantic_fallback"`, `"memory_heuristic"`).
  - `token_count`: Estimated token footprint for context budget accounting.
  - `provenance`: Structured tracking containing document IDs, hashes, turn sequences, and raw scores.
- **`RetrievalQuery`**: Parameter bundle specifying text, `top_k`, `score_threshold`, source filters, and fusion weights (`lexical_weight`, `semantic_weight`).
- **`RetrievalResult`**: Unified response containing ranked `List[RetrievalCandidate]`, scanned count, and wall-clock execution telemetry.

---

## 4. EMBEDDING PROVIDER CONTRACT

To enable dense vector retrieval without binding the codebase to specific heavy frameworks (such as PyTorch Hub, Hugging Face Transformers, or ONNX runtime), Step 13 establishes an abstract interface:

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]: ...
    
    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]: ...
    
    @property
    @abstractmethod
    def dimension(self) -> int: ...
    
    @property
    @abstractmethod
    def provider_id(self) -> str: ...
    
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float: ...
```

### Key Contract Rules:
1. Vectors are standard Python float lists, ensuring zero serialization friction.
2. Cosine similarity computes $S_C = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$, strictly clamped to $[-1.0, 1.0]$, returning $0.0$ for zero-norm inputs.
3. Dimensionality mismatches explicitly raise `ValueError`.

---

## 5. REFERENCE EMBEDDING IMPLEMENTATION

To validate vector generation, indexing, similarity calculation, and provider hot-swapping without introducing external weights or dependencies, Step 13 provides `DeterministicHashEmbeddingProvider`:

### Critical Distinction:
> **ARCHITECTURAL HARNESS, NOT AN INTELLIGENT SEMANTIC MODEL.**  
> The `DeterministicHashEmbeddingProvider` is an architectural test and baseline component. It computes deterministic signed projections over word tokens and character 3-grams into a fixed hash bucket space using SHA-256 and L2 normalization ($d=64$).  
> It does **NOT** possess neural semantic comprehension and must **NOT** be claimed or described as an intelligent embedding model. Its role is to verify deterministic vector indexing, ranking, and score fusion.

---

## 6. IN-MEMORY VECTOR INDEX

ChakrView Step 13 implements `InMemoryVectorIndex`:
- **Zero Heavyweight Dependencies**: No FAISS, Chroma, Pinecone, or C-extension libraries.
- **Exact Cosine Search**: Evaluates query vectors against registered items with $O(N \cdot d)$ brute-force comparison, delivering sub-millisecond latencies for hundreds of documents.
- **Deterministic Tie-Breaking**: When multiple candidates yield identical cosine similarity scores, ties are deterministically broken via alphanumeric sorting on `(-score, item_id)`.
- **Dynamic Lifecycle**: Supports `add_vector`, `search`, `remove`, `count`, and `clear`.

---

## 7. HYBRID SCORE FUSION

`HybridRetriever` combines Okapi BM25 lexical retrieval and dense vector similarity into a single composite ranking.

### Score Normalization & Fusion Formulation

Given:
- $S_{\text{lex}} \ge 0.0$ (Okapi BM25 score, unbounded above)
- $S_{\text{sem}} \in [-1.0, 1.0]$ (Cosine vector similarity)

1. **Normalization**:
   $$\hat{s}_{\text{lex}} = \frac{S_{\text{lex}}}{\max_{c \in \mathcal{C}_{\text{lex}}} (S_{\text{lex}}(c))} \quad (\text{clamped to } [0.0, 1.0])$$
   $$\hat{s}_{\text{sem}} = \max(0.0, S_{\text{sem}}) \quad (\text{non-negative cosine similarity})$$

2. **Weight Normalization**:
   $$w'_{\text{lex}} = \frac{w_{\text{lex}}}{w_{\text{lex}} + w_{\text{sem}}}, \quad w'_{\text{sem}} = \frac{w_{\text{sem}}}{w_{\text{lex}} + w_{\text{sem}}}$$

3. **Composite Fusion Score**:
   $$S_{\text{hybrid}} = w'_{\text{lex}} \cdot \hat{s}_{\text{lex}} + w'_{\text{sem}} \cdot \hat{s}_{\text{sem}}$$

### Subsystem Fallback Guarantees:
- **No Lexical Matches** (e.g. query contains out-of-vocabulary terms): Falls back purely to normalized semantic similarity ($S_{\text{hybrid}} = \hat{s}_{\text{sem}}$, `retrieval_method="semantic_fallback"`).
- **No Semantic Matches** (e.g. empty vector index): Falls back purely to normalized lexical score ($S_{\text{hybrid}} = \hat{s}_{\text{lex}}$, `retrieval_method="lexical_fallback"`).
- **Deterministic Ordering**: Candidates are sorted by `(-S_hybrid, candidate_id)`.

---

## 8. MEMORY & KNOWLEDGE UNIFICATION

`UnifiedRetriever` acts as the orchestration layer coordinating disparate retrieval sources:
- **Architectural Isolation**: Does **NOT** merge the underlying storage models. `WorkingMemory` retains its short-term eviction, session boundaries, and decay parameters in `memory.py`; `KnowledgeIndex` maintains document chunking and indexing in `knowledge.py`.
- **Cross-Source Ranking**: Queries both knowledge sources (via `HybridRetriever`) and working memory (via `WorkingMemory.retrieve_relevant`).
- **Memory Priority Boost**: Applies an optional configurable boost (default $+0.10$) to active short-term conversational working memories to prioritize user-stated immediate constraints over static background knowledge.
- **Unified Output**: Emits a ranked list of `RetrievalCandidate` items labeled with their explicit source type.

---

## 9. CONTEXT INTEGRATION & 512-TOKEN BUDGET HORIZON

`PromptContextBuilder` was updated to consume `unified_candidates`:
1. **Partitioning**: Automatically splits unified candidates into `effective_chunks` (for knowledge grounding) and `effective_memories` (for conversational context).
2. **Context Budget Partitioning**:
   $$\text{System} + \text{Memory} + \text{Knowledge} + \text{History} + \text{Query} + \text{Generation Budget} \le 512 \text{ tokens}$$
3. **Strict Priority Under Contention**:
   1. Current User Query & System Instructions (strictly preserved).
   2. Working Memory (up to `max_memory_tokens`, default 140).
   3. Knowledge Grounding (up to `max_knowledge_tokens`, default 240).
   4. Recent History (oldest turns evicted first, default 100).
   5. Generation Horizon (reserved 64 tokens).

---

## 10. SECURITY & PROMPT-INJECTION CONTAINMENT

Retrieved items (from vector search, BM25, or working memory) are strictly passive data:
- **Unambiguous Delimiters**:
  ```text
  --- WORKING MEMORY START ---
  [PREFERENCE] User prefers concise responses.
  --- WORKING MEMORY END ---

  --- KNOWLEDGE CONTEXT START ---
  [DOCUMENT doc_001_c00] (Score: 0.92, Method: hybrid)
  ChakrMicro is an indigenous neural architecture...
  --- KNOWLEDGE CONTEXT END ---

  --- USER QUERY ---
  What is the parameter count?
  ```
- **Injection Isolation**: If retrieved content contains injection attempts (e.g. `"IGNORE ALL PREVIOUS INSTRUCTIONS"`), it is quarantined within passive data blocks and cannot override system instructions or assume tool execution privileges.
- **Zero Tool Authority**: Retrieved candidates cannot invoke tools; only verified system skills can authorize tool execution.

---

## 11. EMPIRICAL BENCHMARK RESULTS

All benchmarks were measured on a local Intel CPU environment using `scripts/benchmark_hybrid_retrieval.py` and recorded in `docs/STEP_13_BENCHMARK_RESULTS.json`.

### A. Embedding Generation Latency
| Metric | Measured Value |
|---|---|
| Vector Dimension ($d$) | 64 |
| Single Text Latency | **95.15 $\mu$s** |
| Batch 50 Latency | **4.42 ms** |
| Throughput | **10,509 embeddings/sec** |

### B. Retrieval Scaling Across Corpus Sizes
| Corpus Size (Docs) | Vector Insertion (ms) | Vector Search ($\mu$s) | BM25 Search ($\mu$s) | Hybrid Search ($\mu$s) | Hybrid Search (ms) |
|---|---|---|---|---|---|
| **10** | 2.49 ms | 43.50 $\mu$s | 7.87 $\mu$s | 98.73 $\mu$s | **0.099 ms** |
| **100** | 21.17 ms | 443.85 $\mu$s | 71.66 $\mu$s | 582.93 $\mu$s | **0.583 ms** |
| **1,000** | 201.09 ms | 4,869.98 $\mu$s | 746.58 $\mu$s | 6,024.52 $\mu$s | **6.025 ms** |

### C. Working Memory & Context Assembly Latencies
| Component | Workload | Latency |
|---|---|---|
| Working Memory Retrieval | 10 items | 23.34 $\mu$s |
| Working Memory Retrieval | 50 items | 25.68 $\mu$s |
| Working Memory Retrieval | 100 items | 24.25 $\mu$s |
| Prompt Context Assembly | Multi-source unified prompt | **10.64 $\mu$s** |

### D. End-to-End Inference Latency by Mode
| Execution Mode | Description | Latency (ms) | Tokens Generated | Knowledge Used |
|---|---|---|---|---|
| **Mode 1: Base Generation** | Direct inference (no retrieval) | **63.60 ms** | 16 | False |
| **Mode 2: BM25 Lexical RAG** | BM25 retrieval + generation | **111.99 ms** | 16 | True |
| **Mode 3: Semantic Vector RAG** | Dense vector retrieval + generation | **122.88 ms** | 16 | True |
| **Mode 4: Hybrid RAG** | BM25 + Vector fusion + generation | **125.91 ms** | 16 | True |
| **Mode 5: Unified Orchestration** | Working Memory + Hybrid Knowledge + Chat | **143.71 ms** | 16 | True |

---

## 12. ARCHITECTURAL LIMITATIONS

1. **Reference Embedding Provider**: The included `DeterministicHashEmbeddingProvider` is an architectural test harness for vector mathematics, not a semantic understanding model.
2. **In-Memory Index Scale**: `InMemoryVectorIndex` performs exact brute-force cosine search ($O(N \cdot d)$). While extremely fast for $\le 1,000$ documents ($6.0$ ms), million-scale corpora will eventually require approximate nearest neighbor (ANN) indexing.
3. **Linear Score Fusion**: The current linear weighted combination relies on dynamic max-normalization. Highly skewed BM25 distributions with single outlier terms may compress the variance of remaining lexical scores.

---

## 13. FUTURE UPGRADE PATH TO NEURAL EMBEDDINGS

The modular design of `EmbeddingProvider` and `VectorIndex` ensures that real neural embedding models can be introduced seamlessly in future steps without modifying `InferenceSession`, `PromptContextBuilder`, or `ChakrMicro`:
1. **Drop-in Neural Provider**: Implement `NeuralEmbeddingProvider(EmbeddingProvider)` wrapping a small sovereign encoder (e.g. 30M-parameter MiniLM or indigenous encoder) producing $d=256$ or $d=384$ embeddings.
2. **Reciprocal Rank Fusion (RRF)**: Implement an alternative non-linear rank-based fusion strategy ($RRF(d) = \sum \frac{1}{k + r_i(d)}$) alongside linear weighted fusion.
3. **Disk-Backed Vector Persistence**: Implement a memory-mapped binary vector store for persistent cross-session knowledge bases.

---

## 14. RATIFICATION & SUMMARY

- **All 389 tests passed** (363 baseline + 26 new Step 13 tests).
- **Exact ChakrMicro parameters preserved**: `3,443,136`.
- **Exact tokenizer vocabulary preserved**: `4,096`.
- **Exact context ceiling preserved**: `512`.
- **Model remains strictly stateless; runtime manages retrieval**.
