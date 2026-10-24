# ChakrView Experiment 16 v3.1 — Leakage-Reduced Dataset Pipeline

**Status**: Dataset & Contract Verified — Leakage Audited (NO MODEL TRAINING RUN)  
**Location**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/`  
**Date**: 2026-10-22  

---

## 1. Overview and Core Innovations

Exp16 v3.1 resolves the critical question-level leakage discovered in the Exp16 v3 post-implementation audit:
1. **Zero Question-Level Leakage (100% Shared Queries in Train)**:
   - In Train, **100% of unique queries (25/25)** appear with BOTH answerable and abstention labels.
   - Removed query wording cues across all categories:
     - `distractor_resistance`: Removed `"unlisted node"` from abstention queries; both labels use `Auditor: What is the operating state of {target_node}?`.
     - `anaphora_reference_resolution`: Both labels use `Manager: Which cluster completed the assigned task?`.
     - `ambiguity_clarification`: Standardized query on `reboot` for both labels.
     - `constrained_code_synthesis`: Standardized query on `Developer: What is the required function identifier for the routine?`.
     - `missing_context_abstention` & `factual_uncertainty_handling`: Pooled entity namespaces across answerable/abstention examples so queries are identical across labels.
2. **Measured Baselines Drop to Near-Prior Levels**:
   - Train Exact-Query Majority Baseline: **74.00%** (222/300) — identical to the global label prior!
   - Train Template Majority Baseline: **74.00%** (222/300).
   - Train-to-Test Generalization Baseline: **76.67%** (92/120).
3. **Fail-Closed Span Alignment & Candidate Mask**:
   - Strict candidate masking excludes query tokens, assistant prefix, and batch padding.
   - All spans verified against `ChakrTokenizer` (`vocab_4096`).
4. **Human Review Queue**:
   - [`HUMAN_REVIEW_QUEUE_V3_1.md`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/HUMAN_REVIEW_QUEUE_V3_1.md) includes full context, highlighted target demarcation (`>>>target<<<`), explicit spans, and reviewer decision/reason columns. All 480 items are pending.

---

## 2. Dataset Files & Cryptographic SHA-256 Hashes

| Split | File Path | Examples (Ans / Abs) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Train** | `data/train_exp16_v3_1.jsonl` | 300 (222 / 78) | `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14` |
| **Val** | `data/val_exp16_v3_1.jsonl` | 60 (46 / 14) | `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36` |
| **Test** | `data/test_exp16_v3_1.jsonl` | 120 (92 / 28) | `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af` |

---

## 3. Running the Test Suite

```powershell
.venv\Scripts\pytest.exe experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/test_exp16_v3_1_contract.py -v
```
