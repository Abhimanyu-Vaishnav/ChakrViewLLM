# Batch 241: Repository & Previous-Result Audit

**Date**: 2026-10-14  
**Workspace**: `D:\Project\ChakrView`  
**Development Branch**: `dev/v0.6-v0.7-validation`  
**Inspected Commit**: `a9574b4` — `feat(batch231-240): validate cross-attention and multi-hop grounding`

---

## 1. Verification of Canonical Model Invariants & Starting State

| Invariant / Check | Expected State | Verified State | Status |
|---|---|---|---|
| **Git Branch** | `dev/v0.6-v0.7-validation` | `dev/v0.6-v0.7-validation` | **VERIFIED** |
| **Commit at HEAD** | `a9574b4` | `a9574b4` | **VERIFIED** |
| **Canonical Checkpoint Hash** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **INTACT** |
| **Canonical Parameters** | 3,443,136 | 3,443,136 across 6 layers, $d_{model}=192$, vocab 4096, context window 512 | **VERIFIED** |
| **Hardware Detection** | Adaptive / CPU-only environment | Physical: NVIDIA RTX A1000 6GB Laptop GPU + Intel Iris Xe. PyTorch: `2.14.0+cpu` (CPU-only build, CUDA unavailable). | **VERIFIED (CPU ONLY)** |
| **Protected `session.py`** | Working-tree ChatML edit preserved | Tracked with local changes (lines 314–333); uncommitted and unstaged | **PRESERVED** |

---

## 2. Reproduction of Batches 231–240 Cross-Attention & Multi-Hop Grounding

We re-evaluated the reported multi-hop manifest `frozen_multihop_manifest_100.json` (SHA-256: `b70b38a880e307be77d5eaa1df68bd86b977ad67a2f9d13a79e4b1b7e0939016`):
- **Cross-Attention Pointer + TokenTrie (Arm B2)**: Re-evaluated at **78/100 (78.0%) Exact Match** (58 intermediate hits on answerable chains, 20 correct missing-link abstentions, 7 distractor confusions).
- **Frozen Relation Pointer + TokenTrie (Arm A)**: **50/100 (50.0%) Exact Match**.
- **Cross-Attention Pointer Greedy (Arm B1)**: **33/100 (33.0%) Exact Match**.
- **Deterministic Extractor (Arm C)**: **20/100 (20.0%) Exact Match** (succeeded solely on missing-link abstention, completely failed 2-hop composite reasoning).
- **BM25 Lexical Grounding (Arm D)**: **55/100 (55.0%)**.

### Critical Audit Finding on Model vs. Constraint Contribution
- The jump from 33.0% (Greedy) to 78.0% (TokenTrie) indicates that **TokenTrie candidate sequence constraints play a decisive role in span reconstruction**. The cross-attention layer ($N_{heads}=2$, 150,529 parameters) successfully focuses position weights toward the intermediate entity and value span, but unconstrained autoregressive decoding requires candidate prefix trees to prevent subword token drift.

---

## 3. Audit of the 10 Known Interactive Failures

Tracing the 10 reported interactive failures revealed structural bottlenecks in routing, intent classification, and tool dispatch:
1. **Arithmetic (`25 + 37`)**: Classified as `KNOWLEDGE_RETRIEVAL` because of the query prefix `What is`. The retriever found no documents above threshold, defaulting to an ungrounded hallucination or fallback rejection instead of invoking `CalculatorTool`.
2. **Mathematical Constants (`value of pi`)**: Misrouted or extracted with wrong subject (`value` instead of `pi`).
3. **Acronym Definition (`What is an ALU?`)**: Extracted `an` or `ALU` without standard terminology expansion.
4. **Simplification & General Explanations (`Explain CPU in simple words`)**: Routed to `DIRECT_CHAT`, generating unconditioned neural token loops from canonical ChakrMicro's small parameter base.
5. **Coding Program Requests (`Write a Python program...`)**: Routed to generic chat; no code generator template or restricted sandbox validator was engaged.
6. **False-Premise Questions (`capital of Mars`)**: Treated as knowledge seeking; lacked false-premise detection.
7. **Hinglish/Hindi Requests (`Mujhe simple Hindi mein...`)**: Lack of Hinglish intent parser caused routing to default chat.
8. **Contextual Follow-Ups (`and What about ALU`)**: Prior turn entity context was not linked during routing.
9. **User Attribute Memory (`My name is Ramesh` $\to$ `What is my name?`)**: Profile attributes were not stored in persistent memory slots.
10. **Memory Persistence Across Restarts**: Required isolated disk-backed session persistence.

These root causes are addressed in Batches 242–249.
