# Milestone Summary: Batches 191–200
## Query-Aware Pointer Learning, Constrained Decoding, End-to-End Grounded Conversation

**Date:** 2026-10-13  
**Branch:** `dev/v0.6-v0.7-validation`  
**Milestone Scope:** Batches 191 to 200  
**Status:** Complete, Verified, Ready for Git Delivery  

---

## 1. Objectives & Architectural Integrity

The objective of Batches 191–200 was to resolve known pointer attention failure modes (specifically query-entity overlap bias and Hindi BPE continuation fragmentation), formalize the answer verification contract, and demonstrate durable, grounded multi-turn conversation with zero memory leaks across sessions.

### Key Invariants Maintained
- **Canonical Model Weights Untouched:** Canonical checkpoint `training_checkpoint_step_10000.pt` verified with SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` and exactly 3,443,136 unique parameters.
- **Protected User File Untouched:** `chakrview/runtime/session.py` was strictly preserved and never modified.
- **Truthful Component Attribution:** All metrics explicitly distinguish neural autoregressive, neural pointer, constrained Trie, and deterministic extraction upper bounds.

---

## 2. Experimental Outcomes Across Batches

| Batch | Focus Area | Deliverable / Result | Status |
|---|---|---|---|
| **191** | Repository Audit & Failure Reproduction | `reports/initial_audit.md` verified canonical checkpoint hash & parameter counts. | Passed |
| **192** | Query-Entity Disentangled Pointer Attention | Implemented soft dampening (-3.5 logit penalty) on prompt overlap tokens in `ChakrPointerHead`. Authored `reports/query_overlap_diagnostics.json`. | Passed |
| **193** | Relation-Binding Dataset & Split Audit | Generated 22 facts across train/val/test with 0 entity leakage. Authored `reports/relation_binding_data_audit.md`. | Passed |
| **194** | Controlled Query-Aware Pointer Training | Query-aware pointer doubled held-out relation exact match from 16.7% to 33.3%, completely eliminating wrong-entity prompt copying. | Passed |
| **195** | Tokenizer-Aware Constrained Decoding | Implemented `TokenTrie` and `decode_pointer_constrained` in `chakrview/runtime/constrained_decoder.py`. Boosted exact match on candidate spans to 66.7%. | Passed |
| **196** | Evidence-Grounded Answer Verification Contract | Authored `reports/answer_verification_contract.md` defining strict 6-step verification rules. | Passed |
| **197** | Multi-Turn Grounded Conversation Benchmark | All 10 turns passed in `run_multiturn_benchmark.py` covering ingestion, queries, pronoun follow-ups, synthesis, conflicts, and recovery. | Passed |
| **198** | Recovery, Conflicts, and Resource Limits | Evaluated 8 bounded stress tests in `run_recovery_resource_eval.py` (8/8 passed). | Passed |
| **199** | Final Ablation & Regression Evaluation | Compiled component-attributed evaluation across 10 configurations in `reports/final_evaluation.json`. | Passed |
| **200** | Consolidation, Documentation & Git Delivery | Authored manifest, reports, test suite, and delivered to development branch. | Passed |

---

## 3. Component Performance Summary

| Arm / Component | Exact Match | Relation Binding | Notes |
|---|---|---|---|
| **Canonical Base (Vocab-Only)** | 0.0% | 0.0% | Unattributed autoregressive generic generation. |
| **Standard Pointer (Greedy)** | 16.7% | 16.7% | Attracted to prompt entity mentions. |
| **Standard Pointer (Beam w=2)** | 33.3% | 33.3% | Beam search provides marginal benefit when overlap occurs. |
| **Query-Aware Pointer (Greedy)** | **33.3%** | **33.3%** | Eliminates prompt-entity bias; doubles greedy EM. |
| **Constrained Trie Decoding** | **66.7%** | **66.7%** | Guarantees valid BPE byte continuations for Hindi & decimals. |
| **Deterministic Extraction (Oracle)** | 100.0% | 100.0% | Rule-based upper bound with citation provenance. |

---

## 4. Highest-Value Next Milestone

- **Batch 201+ Recommendation:** Integrate learned query-aware pointer representations into the interactive REPL while scaling relation-binding pretraining with synthetic bilingual knowledge graphs.
