# Architecture & Failure Audit: Batches 151–160 Cognitive Routing

**Date:** 2026-10-13  
**Milestone:** Batches 151–160 Unified Cognitive Routing & Grounded Memory  
**Scope:** Runtime Architecture, Subsystem Dependency Mapping, and Failure Boundaries  

---

## 1. Executive Summary

ChakrView has progressed through pretraining (Batches 16–27, 10,000 steps), multi-turn SFT formatting (Batches 143–146), and context-binding diagnostics (Batches 147–150). Empirical testing established that while the 3.44M parameter `ChakrMicro` core reliably adopts ChatML dialogue structure and stops cleanly at `<|im_end|>`, it cannot reliably memorize dense factual records or perform zero-shot arbitrary span extraction from weights alone.

To achieve grounded intelligence without relying on external hosted LLMs or uncalibrated heuristics, ChakrView requires a **Modular Two-Tier Cognitive Architecture**:
1. **Tier 1 (Cognitive Routing & Linguistic Synthesizer):** The neural model handles open-ended conversational fluency, intent clarification, and style.
2. **Tier 2 (Grounded Evidence & Deterministic Extraction):** `HybridRetriever` provides provenanced knowledge chunks, while deterministic validators extract and verify structured slots before returning responses to the user.

---

## 2. Dependency Map & Runtime Data Flow

```
User Turn / Query
       │
       ▼
┌──────────────────────────────────────────────┐
│  CognitiveRouter (Batch 152)                 │
│  - Evaluates intent & route policy           │
│  - Routes: DIRECT_CHAT, KNOWLEDGE_QA,        │
│    SLOT_EXTRACTION, INSUFFICIENT_EVIDENCE    │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Route: DIRECT_CHAT]         [Route: KNOWLEDGE_QA / EXTRACTION]
       │                               │
       │                               ▼
       │                     ┌──────────────────────────────┐
       │                     │  HybridRetriever Adapter     │
       │                     │  - Lexical BM25 (sparse)     │
       │                     │  - Vector Cosine (dense)     │
       │                     │  - Provenance & Chunking     │
       │                     └──────────────┬───────────────┘
       │                                    │
       │                                    ▼
       │                     ┌──────────────────────────────┐
       │                     │  Context Packaging & Guard   │
       │                     │  - Thresholding (> 0.20 score)│
       │                     │  - Context Envelope Building │
       │                     └──────────────┬───────────────┘
       │                                    │
       ├────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────┐
│  InferenceEngine / Neural Core               │
│  - Causal tokenization with ChatML delimiters│
│  - Autoregressive greedy generation          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Deterministic Extraction & Verification     │
│  - Exact span validation                     │
│  - Conflicting evidence detection            │
│  - Abstention on ungrounded claims           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Persistent Memory & Session Adapter         │
│  - Bounded multi-turn dialogue state         │
│  - Isolated tenant/session storage           │
└──────────────────────────────────────────────┘
```

---

## 3. Confirmed Defects and Safety Boundaries

1. **User Change Protection (`chakrview/runtime/session.py`):**
   - The user has an active, uncommitted change in `session.py` introducing `ConversationTemplate` ChatML formatting.
   - **Boundary Enforcement:** We will NOT edit, stage, or revert `session.py`. Instead, all cognitive routing and session adapters are developed as independent, modular components in `chakrview/runtime/cognitive_router.py` and `chakrview/runtime/session_adapter.py`.
2. **Retrieval Calibration Defect:**
   - Raw cosine similarity and BM25 scores are not calibrated probabilities. Treating raw similarity as a probability leads to false positive grounding.
   - **Remedy:** Implement explicit score thresholding (e.g., minimum score $0.20$) and return an explicit `INSUFFICIENT_EVIDENCE` status when candidates fall below the threshold.
3. **Delimiter Subword Representation:**
   - `<|im_start|>` and `<|im_end|>` are subwords, not atomic token IDs. The router must ensure prompt boundaries are preserved without accidental token merges.
4. **Attribution Transparency:**
   - Every response must carry metadata declaring whether the answer originated from neural generation, deterministic extraction, or retrieval fallback.
