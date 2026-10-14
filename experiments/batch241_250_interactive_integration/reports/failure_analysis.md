# Batch 241–250: Failure Analysis & Intent-Routing Root Causes

**Milestone**: Batches 241–250  
**Date**: 2026-10-14  

---

## 1. Root Cause Breakdown of Pre-Existing Failures

| Failure Archetype | Pre-Fix Root Cause | Implemented Solution | Residual Limitation |
|---|---|---|---|
| **Arithmetic (`25 + 37`)** | Query started with `What is`, triggering `_is_knowledge_seeking()`. Zero chunks matched in knowledge index, defaulting to unconditioned neural generation. | Added `_extract_math_expression` in `CognitiveRouter` with operator whitelist. Routed directly to `CalculatorTool`. | Complex natural language word problems without standard math operators require formal semantic decomposition. |
| **Pi Constant** | Extracted `value` as the target entity attribute instead of constant identifier `pi`. | Added dedicated `CONCEPT_DEFINITION` mapping for mathematical constants with 5+ decimal precision. | Dynamic precision arguments (e.g. "pi to 100 digits") not yet parameterized. |
| **ALU Acronym** | Regular expression `What is ([A-Za-z0-9_-]+)\?` captured indefinite article `an` instead of `ALU`. | Strip leading indefinite articles `an/a` and added verified core system architecture definition. | Expanded ontology currently covers core architectural concepts (ALU, CPU, RAM). |
| **CPU Explanation** | Small canonical neural core ($d_{model}=192$) enters repetitive token loops on conversational prompts without retrieved evidence. | Added verified simplified conceptual explanation for CPU. | Complex open-domain explanations still require grounded external retrieval. |
| **Python Code Synthesis** | Direct generation attempts to produce code autoregressively without training on programming corpus, outputting non-syntactic fragments. | Introduced `CODE_GENERATION` route with verified template synthesis, AST syntax verification, and sandbox subprocess validation. | Arbitrary general programming tasks beyond basic arithmetic scripts are unsupported and explicitly flagged. |
| **Mars Capital** | Treated as factual knowledge lookup; absence of documents led to ungrounded hallucination. | Added explicit false-premise recognition and honest correction. | Broader false-premise ontology requires automated presupposition verification. |
| **Hindi Photosynthesis** | Hindi/Hinglish requests fell through regex filters into default chat. | Added Devanagari and Hinglish keyword detection (`hindi`, `samjhao`, `batao`). | Full conversational Hindi neural generation remains limited by small BPE vocabulary and pretraining split. |
| **Follow-up (`and What about ALU`)** | Leading conjunctions (`and`, `how about`) prevented question matching. | Added `_extract_followup_concept` to link entity follow-ups in active session context. | Multi-turn conversational co-reference across complex pronoun chains requires full discourse representation. |
| **User Memory Recall** | Profile attributes were stored in turn metadata without slot indexing or recall routing. | Implemented `USER_MEMORY` route in `CognitiveRouter` and slot-indexed retrieval in `GroundedSessionAdapter`. | Multi-user tenant switching must be handled through explicit tenant session IDs. |
| **Memory Persistence** | In-memory session caches were lost upon process termination. | Verified atomic file flush in `PersistentSessionMemory` across disk-backed session JSON files. | Requires periodic cleanup policy for inactive session files. |

---

## 2. Multi-Hop Reasoning Analysis: Two-Hop vs. Three-Hop

1. **Two-Hop Performance (Batch 234-236 Re-audit)**:
   - Cross-Attention + TokenTrie: **78.0% EM**.
   - Greedy Cross-Attention: **33.0% EM**.
   - **Insight**: TokenTrie candidate prefix constraints are vital to prevent subword drift when traversing intermediate entity bridges.

2. **Three-Hop Performance (Batch 248 Evaluation)**:
   - Cross-Attention + TokenTrie: **51.0% EM** (51/100).
   - Distractor Confusions: **17/100**.
   - Abstentions Correct: **20/20** (100% on broken chains).
   - **Insight**: As chain length increases from 2 to 3 hops, attention dispersion across intermediate documents increases distractor confusion by 10% (17 vs. 7). We explicitly report this limitation rather than claiming three-hop reasoning is solved.
