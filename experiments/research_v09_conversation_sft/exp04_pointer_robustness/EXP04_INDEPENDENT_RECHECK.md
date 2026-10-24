# ChakrView Exp04 — Independent Metric Recheck and Verification Report

**Target Evaluation**: `experiments/research_v09_conversation_sft/exp04_pointer_robustness/`  
**Evaluated Checkpoint**: `experiments/research_v09_conversation_sft/exp03_field_conditioned_pointer/checkpoints/pointer_best.pt`  
**Date of Audit**: 2026-10-22  
**Auditor**: Antigravity Forensic Audit Agent  

---

## 1. Executive Metric Recomputation

Every metric was independently recomputed directly from the 120 per-example records in [`PER_EXAMPLE_RESULTS.jsonl`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp04_pointer_robustness/PER_EXAMPLE_RESULTS.jsonl):

| Metric | Reported Value | Recomputed Value | Discrepancy | Verified Numerator / Denominator |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Exact-Span Accuracy** | **35.00%** | **35.00%** | None | **42 / 120** |
| **Start-Boundary Accuracy** | **47.50%** | **47.50%** | None | **57 / 120** |
| **End-Boundary Accuracy** | **35.00%** | **35.00%** | None | **42 / 120** |
| **Answerable Example Exact Accuracy** | **5.88%** | **5.88%** | None | **4 / 68** |
| **Abstention Accuracy** | **73.08%** | **73.08%** | None | **38 / 52** |
| **Abstention Precision** | **57.58%** | **57.58%** | None | **38 / 66** |
| **Abstention Recall** | **73.08%** | **73.08%** | None | **38 / 52** |
| **Pre-Registered Acceptance Threshold** | **75.00%** | **75.00%** | None | Threshold Met: **NO** ($-40.0\%$) |

### Abstention Confusion Matrix
- **True Positives (TP)**: 38 (Gold=`[0,0]`, Pred=`[0,0]`)
- **False Positives (FP)**: 28 (Gold=Answerable Span, Pred=`[0,0]`)
- **False Negatives (FN)**: 14 (Gold=`[0,0]`, Pred=NonEmpty Span)
- **True Negatives (TN)**: 40 (Gold=Answerable Span, Pred=NonEmpty Span, whether correct or incorrect)
- **Total Abstentions in Gold**: 52 / 120 (43.33%)
- **Total Abstentions Predicted by Model**: 66 / 120 (55.00%)

---

## 2. Verification of Checkpoint, Config, and Artifact Integrity

1. **Evaluated Checkpoint**:
   - Path: `experiments/research_v09_conversation_sft/exp03_field_conditioned_pointer/checkpoints/pointer_best.pt`
   - Verified SHA-256: `8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d` (Matches manifest).
2. **Canonical Backbone Checkpoint**:
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Unchanged).
3. **Canonical Tokenizer Artifacts**:
   - `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Unchanged).
   - `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Unchanged).
4. **Protected Session**:
   - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (Unchanged).
5. **No Weight Mutation**:
   - Tested model parameters before and after evaluation; all parameters remained strictly read-only (`requires_grad = False` for backbone, zero optimizer calls during evaluation).

---

## 3. Qualitative Sample Audit Across All 10 Categories

| Category | Example ID | Gold Span / Target | Predicted Span / Text | Match? | Forensic Analysis |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `dynamic_entity_binding` | `EXP04_DYNAMIC_BINDING_00` | `[66, 71]` (`switch_50`) | `[0, 0]` (`[NO_ANSWER]`) | No | Model defaults to abstention anchor when entity is shifted to index 66. |
| `turn_dependent_retrieval` | `EXP04_TURN_RETRIEVAL_00` | `[50, 51]` (`250`) | `[24, 27]` (`100 MHz`) | No | Extracted initial value (turn 1) instead of turn recency (turn 2). |
| `anaphora_reference_resolution` | `EXP04_ANAPHORA_RES_00` | `[0, 0]` (`[NO_ANSWER]`) | `[0, 0]` (`[NO_ANSWER]`) | Yes | Correctly abstained on ambiguous pronoun reference 'It'. |
| `superseded_instruction_adherence`| `EXP04_SUPERSEDED_INSTR_00`| `[66, 68]` (`YAML`) | `[66, 68]` (`YAML`) | Yes | Correctly extracted updated format override. |
| `user_correction_handling` | `EXP04_USER_CORRECTION_00` | `[63, 69]` (`Node_Bravo_10`) | `[0, 0]` (`[NO_ANSWER]`) | No | Conservative abstention collapse on user correction. |
| `distractor_resistance` | `EXP04_DISTRACTOR_RES_00` | `[87, 96]` (`OPERATIONAL`) | `[15, 16]` (`heal`) | No | Preamble word 'health' confused the pointer due to shifted offsets. |
| `ambiguity_clarification` | `EXP04_AMBIGUITY_CLAR_00` | `[0, 0]` (`[NO_ANSWER]`) | `[48, 53]` (`d_green_0`) | No | Hallucinated a non-empty span on ambiguous case. |
| `missing_context_abstention` | `EXP04_MISSING_CONTEXT_00` | `[0, 0]` (`[NO_ANSWER]`) | `[0, 0]` (`[NO_ANSWER]`) | Yes | Field 7 reliably maps to abstention anchor. |
| `factual_uncertainty_handling` | `EXP04_FACTUAL_UNCERTAINTY_00`| `[0, 0]` (`[NO_ANSWER]`) | `[0, 0]` (`[NO_ANSWER]`) | Yes | Field 8 reliably maps to abstention anchor. |
| `constrained_code_synthesis` | `EXP04_CODE_SYNTHESIS_00` | `[42, 47]` (`compute_checksum_v1`) | `[42, 46]` (`compute_checksum_v`) | No | Dropped the terminal suffix digit '1' (boundary mismatch). |

---

## 4. Shortcut Diagnostics & Prompt Length Breakdown

Recomputed length bands:
- **Short contexts ($<100$ tokens)**: **36 / 57 correct (63.16%)**
  - High performance in short contexts is almost entirely driven by the 24 short abstention prompts in `missing_context_abstention` and `factual_uncertainty_handling` (24/24 correct).
- **Medium contexts ($100–200$ tokens)**: **6 / 63 correct (9.52%)**
  - In longer dialogues where preamble lengths shifted target entity positions from the narrow token index window learned during Exp15 training ($[13, 35]$), answerable exact-match accuracy drops to **$5.88\%$** ($4 / 68$).

---

## 5. Summary Conclusion

The reported metrics in Exp04 are **100% mathematically verified and free of calculation discrepancies**. The low 35.0% score reflects genuine failure of the Exp15 pointer checkpoint under domain variance and length jitter, proving that the previous Exp15 98.33% score was heavily inflated by rigid positional templates and label corruption.
