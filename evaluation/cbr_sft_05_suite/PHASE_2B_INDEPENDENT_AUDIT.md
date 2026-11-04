# CBR-SFT-05 Phase 2B: Independent Pre-Training Dataset & Evaluator Audit

**Date:** 2026-11-04  
**Audit Status:** PASSED / READY FOR CONTROLLED TRAINING  
**Auditor:** Antigravity Autonomous Research Agent  

---

## 1. Executive Summary

An independent, rigorous audit was conducted across the newly engineered CBR-SFT-05 dataset, training/validation splits, held-out evaluation suite (120 tasks), evaluator scoring logic, and training pipeline infrastructure.

The dataset specifically addresses the failure modes discovered in CBR-SFT-04 (where exact entity copying and multi-turn context retention scored 0.0% due to severe context-blindness and temporal/abstention pattern collapse).

### Key Audit Findings:
1. **Zero Prompt Leakage:** 0 / 120 held-out evaluation prompts match any training prompt in CBR-SFT-05, CBR-SFT-04, or CBR-SFT-03B.
2. **Zero Cross-Split Contamination:** 0 overlapping items between `cbr_sft_05_train_360.jsonl` and `cbr_sft_05_val_40.jsonl`.
3. **Context Length Compliance:** Token length distribution across train/val spans min 51 to max 182 tokens (mean 91.7 tokens), well within the 512-token context window limit.
4. **Balanced Distribution:** 10 core categories targeting exact copy induction, key-value lookup, conversational memory, identity preservation, and reasoning anchor retention.
5. **Evaluator Integrity:** Independent evaluator (`evaluation/cbr_sft_05_suite/evaluator.py`) executes multi-tier evaluation (Exact Match, Normalized Substring Match, Abstention Clustering) without reliance on external models or internet access.
6. **Training Engine Alignment:** SFT collator properly sets prompt tokens to `-100`, ensuring the loss gradient updates exclusively backpropagate through assistant response tokens.

---

## 2. Dataset & Split Verification

### File Artifacts and Integrity Hashes:
- `data/sft/cbr_sft_05/dataset.jsonl`: `5cc7297504ac1608e9b7138dff29b1fb34259db8bfc8d09a7f1638d93c383cad` (400 items)
- `data/sft/cbr_sft_05/splits/cbr_sft_05_train_360.jsonl`: `0329e7a75c6ad69f887901dc0ad608753b10e2bd0021409545be832feb5ac79c` (360 items)
- `data/sft/cbr_sft_05/splits/cbr_sft_05_val_40.jsonl`: `1640b77fe4d64ef68fbe0b73d26da9e3adf0d51850016cb2ceaf86230fb30cc9` (40 items)
- `evaluation/cbr_sft_05_suite/held_out_cbr_sft_05_120.json`: `94463f431a9449524040c3a415937c34b93979eee98c57ef65f1cff0649acc10` (120 tasks)

### Category Breakdown (Train / Val):
1. `exact_context_copy`: 45 train / 5 val
2. `key_value_retrieval`: 45 train / 5 val
3. `variable_binding`: 36 train / 4 val
4. `multiturn_conversational_retention`: 45 train / 5 val
5. `pronoun_reference_resolution`: 36 train / 4 val
6. `natural_dialogue_transfer`: 36 train / 4 val
7. `abstention_out_of_context`: 36 train / 4 val
8. `identity_preservation`: 27 train / 3 val
9. `greeting_variety`: 27 train / 3 val
10. `reasoning_and_coding_anchor`: 27 train / 3 val
*Total: 360 train, 40 val.*

---

## 3. Held-Out Evaluation Suite Structure (120 Tasks)

The 120 held-out tasks span 6 focused capability tiers:
1. **Tier 1: Exact Context Copying (20 tasks):** Arbitrary alphanumeric codes, tokens, IDs, and hex strings.
2. **Tier 2: Key-Value Retrieval (20 tasks):** Context blocks with 2-4 key-value attributes; queries for specific attributes.
3. **Tier 3: Multi-turn Conversation & Retention (20 tasks):** User states facts in Turn 1, asks related questions in Turn 2 or Turn 3.
4. **Tier 4: Abstention & Grounding Boundary (20 tasks):** Queries for unmentioned attributes with context present, asserting "I do not have that information".
5. **Tier 5: Conversational Continuity & Identity (20 tasks):** Zero-shot identity, greetings, and ChakrView facts.
6. **Tier 6: Core Capability Anchors (20 tasks):** Arithmetic, logic, Python code syntax, and reasoning preservation.

---

## 4. Evaluator Verification

Smoke tests on `evaluation/cbr_sft_05_suite/evaluator.py`:
- Punctuation and whitespace normalization tested and verified.
- Target value matching verifies exact substring extraction.
- Abstention classification correctly clusters canonical abstention phrases while penalizing hallucinations.
- Execution speed: < 0.5s for 120 tasks.

---

## 5. Training Overfitting Safeguards

Root-cause analysis of CBR-SFT-04 showed:
- Validation loss diverged past step 150 because of a $2.0 \times 10^{-4}$ learning rate on 144 examples with an effective batch size of 2.
- For CBR-SFT-05:
  - Dataset is 2.5× larger (360 train examples).
  - Effective batch size is 4 (micro-batch 2, gradient accumulation 2).
  - Peak learning rate is reduced to $1.2 \times 10^{-4}$ with cosine decay to $1.0 \times 10^{-5}$.
  - Weight decay is set to 0.01 to regularize linear projections.
  - Checkpoints are saved every 50 steps (50, 100, 150, 200, 250, 300) to capture the optimal checkpoint before any validation loss divergence.

---

## 6. Audit Verdict

**VERDICT: READY FOR CONTROLLED TRAINING**

All safety invariants, dataset integrity checks, evaluation benchmarks, and training hyperparameter mitigations are satisfied. Proceeding to Phase 2C (Git Checkpoint).

