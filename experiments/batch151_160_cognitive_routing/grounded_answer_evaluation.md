# Batch 154 — Grounded Answer Pipeline Evaluation
## Neural Core Conditioning, Context Demarcation & Component Attribution

### 1. The Grounded Pipeline Architecture
The end-to-end grounded query flow operates via strict component demarcation:
```
User Query
    │
    ▼
Cognitive Router (classify intent, context presence, and factual requirements)
    │
    ├── Route 1: DIRECT_CHAT ───────────► Neural Core Generation (raw conversational dialogue)
    ├── Route 2: INSUFFICIENT_EVIDENCE ──► Fallback Policy ("I do not have sufficient information...")
    └── Route 3: KNOWLEDGE_RETRIEVAL / CONTEXT_QA
            │
            ├──► Hybrid Retriever (search index, filter by threshold > 0.20, attach SHA-256)
            │
            ├── If Structured Slot Sought ──► Evidence Extractor (Exact match with span provenance)
            │
            └── If Open-Ended Context QA ───► Neural Core conditioned on bounded ChatML context
```

### 2. Neural Generation vs. Deterministic Attribution
In ChakrView, **a correct answer from a deterministic extractor is never claimed as neural comprehension.**
The system returns a typed `GroundedResponse` recording:
- `component_origin`: Explicitly marks `neural_core`, `deterministic_extractor`, or `fallback_policy`.
- `is_supported`: Boolean flag indicating whether the answer has traceable evidence in the context envelope.
- `support_explanation`: Textual explanation detailing the exact supporting span or generation regime.
- `retrieved_chunks`: Full list of source chunks with IDs, scores, and provenance hashes.

### 3. Quantitative Evaluation (Batch 151–160 Benchmark)
Across the 15 end-to-end benchmark scenarios:

| Category | Queries Tested | Component Responsible | Success Rate | Provenance Traceability |
|---|---|---|---|---|
| Open Conversational Dialogue | TC-01, TC-10, TC-11, TC-13 | `neural_core` | 100% (4/4) | Conditioned on dialogue turns |
| Grounded Factual QA (Retrieved) | TC-02, TC-03, TC-08, TC-09 | `deterministic_extractor` | 100% (4/4) | Exact supporting span verified |
| Grounded Factual QA (Supplied) | TC-05, TC-06, TC-14 | `deterministic_extractor` | 100% (3/3) | Bounded span in user context |
| Abstention / Missing Facts | TC-04, TC-07, TC-15 | `fallback_policy` / Extractor | 100% (3/3) | Clean, honest abstention |
| Conflicting Evidence Detection | TC-12 | `deterministic_extractor` | 100% (1/1) | Conflicting values flagged |

### 4. Observations on Neural Core Copying
As documented in Batches 147–150, the 3.44M-parameter canonical neural core (`step_10000.pt`) trained on foundational causal language modeling has not yet acquired reliable zero-shot extractive copy behavior. When conditioned on complex context spans, it generates repetitive or token-frequency biased text. 
The cognitive routing layer addresses this limitation safely by routing structured slot extraction and factual attribute queries to the deterministic evidence layer, preserving 100% accuracy while keeping the neural core active for open-ended conversational continuity.
