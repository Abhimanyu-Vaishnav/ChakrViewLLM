# Cognitive Routing Contract Specification: Batch 152

**Date:** 2026-10-13  
**Milestone:** Batch 152 Cognitive Routing Contract  
**Location:** `chakrview/runtime/cognitive_router.py`  

---

## 1. Specification Overview

The `CognitiveRouter` provides an explicit, typed, and observable routing interface that governs how queries traverse ChakrView's neural, retrieval, and deterministic layers.

### Route Types:
1. `DIRECT_CHAT`: General dialogue, greetings, and conversational follow-ups. Handled purely by the neural core using ChatML formatting.
2. `CONTEXT_QA`: Reading comprehension where the user provides explicit context in the prompt envelope.
3. `KNOWLEDGE_RETRIEVAL`: Factual inquiries requiring external lookup in `HybridRetriever`.
4. `SLOT_EXTRACTION`: Attribute-specific or key-value extraction where deterministic parsing can verify the answer against retrieved evidence.
5. `INSUFFICIENT_EVIDENCE`: Fallback triggered when the retriever's confidence falls below threshold ($0.20$), preventing ungrounded hallucinations.

---

## 2. Decision Record Schema

Every route evaluation produces a `RouteDecision` containing:
- `route`: The target execution path (`RouteType`).
- `reason`: Human-readable justification for the decision.
- `confidence`: Confidence score in $[0.0, 1.0]$.
- `decision_method`: Categorization (`policy_rule`, `heuristic`, or `neural_proposal`).
- `requires_retrieval`: Boolean flag indicating whether the knowledge index must be queried.
- `requires_slot_extraction`: Boolean flag indicating whether deterministic verification applies.

---

## 3. Component Attribution Standard

All responses wrap their answer in a `GroundedResponse` data structure that explicitly tags the `component_origin`:
- `NEURAL_CORE`: Raw autoregressive generation from `ChakrMicro`.
- `DETERMINISTIC_EXTRACTOR`: Verified slot extraction from context.
- `HYBRID_RETRIEVER`: Direct chunk or fact match from retrieval.
- `FALLBACK_POLICY`: Standardized abstention response (`"I do not have sufficient information in my knowledge base to answer this question."`).
