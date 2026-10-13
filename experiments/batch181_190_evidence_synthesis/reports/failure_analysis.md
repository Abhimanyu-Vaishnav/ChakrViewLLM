# Failure Analysis & Architectural Diagnostics: Batches 181–190

## 1. Executive Summary
Batches 181–190 advanced ChakrView from single-chunk greedy copy operations into:
1. **Bounded Beam Search (`decode_pointer_beam`):** Parallel hypothesis tracking over multi-token sequences with length normalization.
2. **Two-Stage Hybrid Retrieval Re-Ranking (`HybridReranker`):** Candidate pool generation via BM25 followed by learned bi-encoder re-ranking.
3. **Structured Multi-Document Evidence Assembly (`MultiDocumentEvidenceAssembler`):** Content hash deduplication, cross-document contradiction detection, and context budget management ($\le 380$ tokens).
4. **Multi-Document Synthesis in `CognitiveTaskEngine`:** Durable multi-step workflows with restart recovery and claim-level provenance.

---

## 2. Pointer Decoding: Greedy vs Beam Analysis
### 2.1 Empirical Findings on Held-Out Test Facts
| Decoding Mode | Beam Width ($k$) | Held-Out Exact Match | Latency per Case | Failure Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Vocab-Only** | 1 | 20.0% | 191.35 ms | Repetitive wiki formatting table tokens |
| **Pointer Greedy** | 1 | 40.0% | 185.12 ms | Myopic first-token choices; multi-token fragmentation |
| **Pointer Beam ($k=2$)** | 2 | 40.0% | 462.19 ms | Maintains top hypotheses; recovers ordered tokens |
| **Pointer Beam ($k=3$)** | 3 | 40.0% | 736.82 ms | Robust against rare prefix tokens; higher compute cost |
| **Deterministic Oracle** | N/A | **80.0%** | **0.05 ms** | Strict span extraction reference upper bound |

### 2.2 Why Beam Search Preserved (Rather than Doubled) Factual EM on 5 Cases
The limiting factor in the 60% failing cases (`Neptune -> Triton`, `Gold -> 1064 C`, `चांदी -> विद्युत सुचालक`) was not beam pruning, but:
1. **Query Overlap Bias:** The pointer attention distribution $P_{\text{copy}}$ assigned probability to words appearing in both query and context (e.g. `Neptune`) instead of shifting to the relation object (`Triton`).
2. **Tokenizer Boundary Discrepancies:** For Hindi multibyte tokens (`विद्युत`), byte-level BPE requires multi-step continuation tokens that the small auxiliary head has not yet aligned.

---

## 3. Hybrid Retrieval: BM25 Candidate Generation + Neural Re-Ranking
### 3.1 Two-Stage Pipeline Results
| Retrieval Pipeline | Recall@1 | Recall@3 | Mean Reciprocal Rank (MRR) | Latency per Query |
| :--- | :--- | :--- | :--- | :--- |
| **BM25 Lexical Alone** | **100.0%** | **100.0%** | **1.0000** | **0.03 ms** |
| **BM25 + Hash Rerank** | **100.0%** | **100.0%** | **1.0000** | 0.21 ms |
| **BM25 + Neural Bi-Encoder**| **100.0%** | **100.0%** | **1.0000** | 8.84 ms |

### 3.2 Architectural Invariant
Because BM25 candidate generation provides 100% recall on exact keyword matches, learned bi-encoder re-ranking preserves top-1 precision while adding semantic similarity scores. However, BM25 alone is 290x faster on CPU. Therefore, lexical retrieval remains the primary safe fallback when compute is constrained.

---

## 4. Multi-Document Contradiction and Deduplication
1. **Content Hash Deduplication:** Exact identical chunks across documents are pruned to avoid wasting the model's 512-token context window.
2. **Conflict Abstention:** When Document A asserts `boiling point is 105 C` and Document B asserts `boiling point is 115 C`, the engine flags `has_conflicts=True` and safely abstains rather than producing a hallucinated compromise.

---

## 5. Adversarial Evaluation Battery
The 20-dimension regression battery in `run_integrated_evaluation.py` passed **20/20 test cases (100.0%)** in 0.58 seconds.
