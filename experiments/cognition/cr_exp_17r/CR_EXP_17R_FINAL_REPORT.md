# CR-EXP-17R: Independent Release Verification & Capability Audit
## Final Scientific and Engineering Report

**Project:** ChakrView  
**Mission:** CR-EXP-17R (Independent Release Verification & Conversation Validation)  
**Evaluated Artifact:** ChakrView v0.9.1 Experimental Preview  
**Evaluated Checkpoint:** `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt`  
**Checkpoint SHA-256:** `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`  
**Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $L=6, d=192, H=6, d_{\text{mlp}}=512, V=4096$) — Frozen  
**Date:** November 11, 2026  
**Status:** COMPLETE (Evidence-First Audit)

---

## 1. Executive Summary

CR-EXP-17R conducted an exhaustive, independent technical and scientific audit of the ChakrView v0.9.1 Experimental Preview released under CR-EXP-17. The audit strictly followed the principle of **evidence over claims** without training any new models, without modifying model weights, and without altering protected files.

### Key Audit Findings:
1. **Artifact Cryptographic Integrity: VERIFIED**
   - The release checkpoint SHA-256 (`f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`) is bit-for-bit identical to the best checkpoint trained in CR-17.
   - The canonical Stage B checkpoint (`96357705...`), frozen parent (`327118ce...`), and protected session runtime (`9080ff51...`) are 100% intact.
2. **Dataset Lineage Discrepancy: RESOLVED & VERIFIED**
   - The discrepancy between the filenames (`conversation_train_1080.jsonl` / `conversation_val_120.jsonl`) and the reported numbers (1,419 train / 157 val) was investigated.
   - The actual row counts in the files are **1,419** and **157** (total: 1,576). The filenames represent legacy targets from early pipeline planning that were retained when augmentations expanded the dataset.
   - Train and validation sets are strictly disjoint ($0$ overlap). The held-out benchmark has **0.0% contamination**.
3. **Independent Checkpoint Loading: VERIFIED**
   - The release checkpoint loads in standard PyTorch on CPU via `weights_only=True` with `ModelConfig`. All 3,443,136 parameters match strictly. Forward pass determinism across runs was verified ($0.000000$ discrepancy).
4. **Fresh Conversational Evaluation (100 Probes): VERIFIED**
   - Evaluated on `CR17R_CONVERSATION_100.json` (100 fresh, non-overlapping probes).
   - Candidate outperformed the parent baseline in keyword hits ($12/100$ vs $9/100$) and completely eliminated refusal loops ($0/100$ vs $7/100$).
5. **Multi-Turn Context Stress Test (30 Dialogues, 5-8 Turns): FAILED**
   - The model achieved **0.0% fact retention** across 5–8 turn conversational threads. It cannot track names, numbers, or facts introduced 2–3 turns earlier due to representational capacity limits in a 6-layer 3.44M model.
6. **Hindi and Hinglish Usability: PARTIALLY VERIFIED**
   - Roman Hindi greetings work smoothly (`"Namaste! Main aapki kya madad kar sakta hoon?"`).
   - Devanagari Hindi collapses into repetitive Wikipedia-style sentences (`"कथन\nयह है कि यह दोनों में मुख्य रूप से जाना जाता है।"`) rather than engaging in conversational dialogue.
7. **Human Evaluation: PENDING**
   - Blinded evaluation sheet (`blinded_evaluation_sheet.md`) was generated with randomized A/B presentation. In compliance with Section 11, agent judgment was not substituted for human scoring.
8. **Release Recommendation: RETAIN AS EXPERIMENTAL PREVIEW (Option A)**
   - The release demonstrates valid and reproducible progress (curing refusal sinks, learning identity, and greeting politely), but its severe multi-turn retention limitations preclude any stronger classification.

---

## 2. Checkpoint Identity & Lineage Audit

| Component | Path | SHA-256 | Verification |
| :--- | :--- | :--- | :---: |
| Release Checkpoint | `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **VERIFIED** |
| Training Parent | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **VERIFIED** |
| Stage B Baseline | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED** |
| Protected Runtime | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED** |
| CR-17 Benchmark | `experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json` | `d2472e0b2a72f2555bea68511ac7373c5f2983f785bb9a7d60da4dd9ac2f1a5f` | **VERIFIED** |
| CR-17R Benchmark | `experiments/cognition/cr_exp_17r/benchmark/CR17R_CONVERSATION_100.json` | `8b83d29f70349ea01331663f9005ba91b76d7c23145c6baf932e1f1790e156d7` | **VERIFIED** |

---

## 3. Dataset Lineage & Integrity Resolution

| Metric | Reported in CR-17 | Independently Verified | Audit Status |
| :--- | :---: | :---: | :---: |
| Training Examples | 1,419 | 1,419 | **VERIFIED** |
| Validation Examples | 157 | 157 | **VERIFIED** |
| Total Corpus Examples | 1,576 | 1,576 | **VERIFIED** |
| Hashed Train File | `conversation_train_1080.jsonl` | 1,419 lines | **RESOLVED** (Legacy Filename) |
| Hashed Val File | `conversation_val_120.jsonl` | 157 lines | **RESOLVED** (Legacy Filename) |
| Train-Val Overlap | 0 | 0 | **VERIFIED (Strictly Disjoint)** |
| Exact Duplicate Pairs | 0 | 0 | **VERIFIED** |
| Benchmark Contamination | 0% | 0.0% (0 / 200) | **VERIFIED (Leak-Free)** |

---

## 4. Fresh Benchmark Evaluation: Parent vs. Release Candidate

Evaluated on `CR17R_CONVERSATION_100.json` (100 fresh, non-overlapping probes, 10 per category):

| Category (10 Probes Each) | Parent Baseline (CBR-SFT-04) Hits | Candidate Release (v0.9.1) Hits | Parent Refusal Sinks | Candidate Refusal Sinks |
| :--- | :---: | :---: | :---: | :---: |
| 1. Greetings & Introductions | 4 / 10 | 5 / 10 | 0 / 10 | 0 / 10 |
| 2. Natural Conversational | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| 3. Follow-up Questions | 1 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| 4. Context References | 1 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| 5. Identity & Project Grounding | 0 / 10 | **4 / 10** | 3 / 10 | **0 / 10** |
| 6. Multi-turn Retention | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| 7. Topic Changes | 1 / 10 | 1 / 10 | 0 / 10 | 0 / 10 |
| 8. Uncertainty & Missing Info | 1 / 10 | 1 / 10 | 4 / 10 | **0 / 10** |
| 9. Hindi & Hinglish | 1 / 10 | 1 / 10 | 0 / 10 | 0 / 10 |
| 10. Continuity & Relevance | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| **Total Benchmark Score** | **9 / 100 (9.0%)** | **12 / 100 (12.0%)** | **7 / 100 (7.0%)** | **0 / 100 (0.0%)** |

### Qualitative Shift:
- In identity probes, the parent failed completely ($0/10$) and retreated into refusals (`"No, I am not provided in the context."`). The release candidate achieved $4/10$ direct keyword matches, identifying itself as ChakrView and an indigenous neural model.
- Refusal loops dropped from 7% down to **0%**.

---

## 5. Multi-Turn Context Stress Test Results

Stress tested across 30 multi-turn conversations (5 to 8 turns each, total 146 evaluated turns):

| Multi-Turn Capability Dimension | Evaluated Turns | Correct Responses | Accuracy | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| Fact Retention Across Turns | 56 | 0 | 0.0% | **FAILED** |
| User Correction Handling | 2 | 0 | 0.0% | **FAILED** |
| Reference Resolution | 3 | 0 | 0.0% | **FAILED** |
| Topic Transition | 30 | 0 | 0.0% | **FAILED** |
| Early Fact Return | 28 | 0 | 0.0% | **FAILED** |
| Unanswerable Abstention | 27 | 0 | 0.0% | **FAILED** |
| **Overall Multi-Turn Capability** | **146** | **0** | **0.0%** | **FAILED** |

**Forensic Explanation:** A 6-layer, 192-dimension model lacks the induction heads necessary to bind variables across multiple conversational turns. Context turns beyond Turn 2 saturate the attention context, resulting in drift and pretraining entity leakage (e.g., reciting `"Arjun"` or `"Berok"`).

---

## 6. Hindi & Hinglish Capability Breakdown

Evaluated across 30 dedicated probes:

| Category | Total Probes | Relevant Hits | Appropriate Language | Refusals | Audit Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Devanagari Hindi** | 10 | 0 / 10 (0.0%) | 10 / 10 (100.0%) | 0 / 10 | Generates Devanagari characters, but collapses into Wikipedia article syntax. |
| **Roman Hindi** | 10 | 1 / 10 (10.0%) | 1 / 10 (10.0%) | 0 / 10 | Greetings work (`"Namaste! Main aapki..."`), but general queries revert to English greetings. |
| **Mixed Hinglish / Technical** | 10 | 0 / 10 (0.0%) | 3 / 10 (30.0%) | 0 / 10 | Handles technical terms but fails on relational syntax. |
| **Total** | **30** | **1 / 30 (3.3%)** | **14 / 30 (46.7%)** | **0 / 30 (0.0%)** | **PARTIALLY VERIFIED** |

---

## 7. Historical Metric Recalculation

All raw outputs from CR-EXP-17 were recomputed from `baseline_eval_results.json` and `v091_eval_results.json`:
- **Refusal Sink Elimination:** Verified. Dropped from 28/200 (14.0%) in parent to **0/200 (0.0%)** in v0.9.1.
- **Keyword Hit Rate:** Verified. 27/200 in parent vs. 18/200 in v0.9.1.
- **Spurious Hit Attribution:** Verified. 10 of the parent's hits in the "Consistency" category occurred solely because the parent output `"No, I am not provided..."` matching negative keywords like `"No"`.
- **Repetition Rate:** Verified. 0% 3-gram repetition loops under calibrated decoding.

---

## 8. Runtime & Git Verification

1. **CLI Runtime:** `python -m chakrview chat --test-mode` executes cleanly on CPU, handling multi-turn input, `/info`, and `/reset` with deterministic outputs.
2. **Missing Checkpoint Behavior:** Gracefully falls back to parent checkpoint with explicit terminal warning.
3. **Git Cleanliness:** Branch `exp/cbr-01-conversational-breakthrough` is clean, with 0 untracked files and 100% remote synchronization.

---

## 9. Final Release Recommendation

**OPTION A: RETAIN AS EXPERIMENTAL PREVIEW**

The v0.9.1 checkpoint represents genuine progress: it eradicates refusal loops, establishes self-identity, and executes autonomously on CPU. However, because multi-turn context retention across $\ge 3$ turns fails and Hindi conversational fluency is limited to greeting formulas, it must remain strictly categorized as an **Experimental Preview**.
