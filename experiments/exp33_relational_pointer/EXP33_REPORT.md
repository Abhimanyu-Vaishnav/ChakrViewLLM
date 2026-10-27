# EXPERIMENT 33 REPORT: RELATIONAL POINTER CIRCUIT, KEY-TO-VALUE BINDING & INDEPENDENT VALIDATION

**Date:** 2026-10-27
**Status:** COMPLETE — EMPIRICALLY AUDITED & VALIDATED
**Investigator / Auditor:** Principal ML Research Engineer & Repository Safety Owner
**Repository:** `ChakrView` (`Abhimanyu-Vaishnav/ChakrViewLLM`)
**Base Commit:** `bcd67d97af5d826a481c706c1f23cd4b6840ad15`
**Preregistration Hash (SHA-256):** `f46862490cdbf9f63b9a93f3e532818dc168ea4d3e8e7b63f9c3971767078d7c`

---

## 1. Executive Summary & Verdict

### Classification: **FAIL (Against Production Promotion Gate) / PASS (Scientific Diagnostic Integrity)**
- **Audit of Exp32:** Complete and verified. Recomputation of confusion matrices and distractor frequencies confirmed that distractor selection accounted for $99.02\%$ of non-abstention errors in Arm 1 and $100\%$ in Arm 2.
- **Relational Pointer Circuit Evaluation (Arm 2):** Explicit key-to-query matching combined with a key-to-value transition kernel was implemented and trained under an identical 160-example budget (300 steps).
- **Core Empirical Finding:**
  - Arm 2 achieved $24/50$ ($48.0\%$) on slot permutations (highest in the study).
  - However, on 4-key distractors (Group D), Arm 2 achieved only $11/50$ ($22.0\%$), compared to $14/50$ ($28.0\%$) for standard pointer Arm 1.
  - Overall accuracy across 350 test instances was $101/350$ ($28.86\%$) for Arm 2 vs $95/350$ ($27.14\%$) for Arm 1.
- **Hypothesis Disproof ($H_1$ Falsified):** Explicit bilinear key-matching without deep multi-layer transformer routing did not eliminate distractor slot confusion ($39/50$ distractor errors on Group D). Distractor confusion in small transformers cannot be resolved by an isolated linear cross-attention head alone.
- **Milestone Recommendation:** **REJECT for canonical model integration.** Do not merge pointer or relational heads into `main`. The canonical model remains preserved and unchanged.

---

## 2. Research Question & Hypotheses

### Primary Research Question
Does explicitly modeling key-to-query matching before routing pointer attention to associated value tokens (`ChakrRelationalPointerMicro`) resolve distractor slot misattributions and improve retrieval accuracy across variable-order contexts and unseen tokens, while maintaining reliable abstention on absent keys?

### Pre-registered Hypotheses Outcomes
- **$H_0$ (Null Hypothesis): Supported.** Distractor retrieval accuracy remained low ($22.0\% \le 26.0\%$), and distractor errors accounted for $100\%$ of retrieval mistakes ($39/39$ on Group D).
- **$H_1$ (Relational Distractor Reduction $\ge 40\%$): Falsified.** Group D accuracy was $22.0\%$, and distractor errors did not decrease.
- **$H_2$ (Permutation Generalization $\ge 50\%$): Partially Supported.** Permutation accuracy reached $48.0\%$ ($24/50$), confirming that relational alignment makes slot order largely invariant, but fell just short of the $50\%$ gate.
- **$H_3$ (Abstention F1 $\ge 0.50$): Falsified.** Arm 2 achieved an abstention F1 of $0.3377$ (Precision: $48.15\%$, Recall: $26.00\%$), similar to Exp32 ($0.3514$).

---

## 3. Evaluated Model Variants & Execution Details

### Model Architectures & Parameter Counts
1. **Arm 0 (Canonical Zero-Shot Baseline):**
   - Step 10,000 checkpoint (`96357705...`), frozen weights, greedy decoding.
   - Parameters: 3,443,136.
2. **Arm 1 (Standard Pointer Baseline - ChakrPointerMicro):**
   - Core `ChakrMicro` (3.44M) + single-head dot-product pointer head + $p_{gen}$ copy gate.
   - Parameters: 3,480,385 (+37,249 params).
   - Training loss: $7.0107 \to 0.0727$ (Val loss: $0.9854$, wall-clock: 11.9s).
3. **Arm 2 (Relational Pointer Circuit - ChakrRelationalPointerMicro):**
   - Core `ChakrMicro` (3.44M) + two-stage relational head ($W_q^{rel}, W_k^{rel}, W_v^{proj}, W_{key\_to\_val}$) + abstention gate.
   - Parameters: 3,492,866 (+49,730 params).
   - Training loss: $7.7139 \to 0.1082$ (Val loss: $0.9579$, wall-clock: 12.9s).

---

## 4. Empirical Evaluation Results

### Cross-Arm Comparison on 350 Frozen Test Instances

| Evaluation Group | Arm 0: Canonical Zero-Shot | Arm 1: Standard Pointer | Arm 2: Relational Pointer Circuit |
|---|:---:|:---:|:---:|
| **Group A (Familiar In-Distribution)** | 1 / 50 (2.0%) | **17 / 50 (34.0%)** | 16 / 50 (32.0%) |
| **Group B (Unseen Token Identities)** | 0 / 50 (0.0%) | **15 / 50 (30.0%)** | 13 / 50 (26.0%) |
| **Group C (Slot Permutations)** | 6 / 50 (12.0%) | 22 / 50 (44.0%) | **24 / 50 (48.0%)** |
| **Group D (4-Key Distractors)** | 1 / 50 (2.0%) | **14 / 50 (28.0%)** | 11 / 50 (22.0%) |
| **Group E (Counterfactual Dynamic Pairs)** | 7 / 100 (7.0%) | 21 / 100 (21.0%) | **24 / 100 (24.0%)** |
| **Group F (Missing-Key Abstention)** | 0 / 50 (0.0%) | 6 / 50 (12.0%) | **13 / 50 (26.0%)** |
| **Overall Exact Match** | **15 / 350 (4.29%)** | **95 / 350 (27.14%)** | **101 / 350 (28.86%)** |

### Error Distribution & Distractor Breakdown (300 Retrieval Instances)
- **Arm 1 (Standard Pointer):**
  - Retrieval Errors: 211
  - Distractor Selections: 204 ($96.68\%$)
  - False Abstentions: 6 ($2.84\%$)
  - Hallucinations: 1 ($0.47\%$)
- **Arm 2 (Relational Pointer Circuit):**
  - Retrieval Errors: 212
  - Distractor Selections: 198 ($93.40\%$)
  - False Abstentions: 14 ($6.60\%$)
  - Hallucinations: 0 ($0.00\%$)

---

## 5. Failure Analysis: The Mechanistic Limit of Shifting Kernels

Why did the Relational Pointer Circuit not resolve distractor confusion on Group D?
1. **Context Overcrowding:** In Group D, four key-value pairs are compressed into the sequence: `alpha`, `beta`, `gamma`, `delta`.
2. **Representation Entanglement:** In a 6-layer 192-width model, the contextual hidden state at each value position already absorbs substantial attention from adjacent key and stream tokens. A single linear shift kernel $W_{key\_to\_val}(h_{key})$ cannot cleanly unmix which value token is syntactically attached to which key when all four keys are present in a short sequence.
3. **Implication:** The bottleneck is not simply the pointer readout head; it is the **backbone representation's binding capacity**. At 3.44M parameters with $d_{model}=192$, the representations of value tokens in multi-key sequences remain partially entangled.

---

## 6. Regression & Artifact Integrity

1. **Canonical Checkpoint Hash (`training_checkpoint_step_10000.pt`):**
   `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**Bit-exact match**)
2. **Protected Model Core (`chakrview/brain/model.py`):**
   `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (**Bit-exact match**)
3. **Protected Runtime Session (`chakrview/runtime/session.py`):**
   `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**Bit-exact match**)
4. **Canonical Tokenizer & Merges:** All bit-exact matches against release manifests.
5. **Regression Smoke Tests:** `pytest -q tests/test_v060_release_smoke.py` $\to$ **7 passed in 2.29s (100% PASS)**.
6. **Git Hygiene:** `git diff --check` passed cleanly with 0 formatting/whitespace errors.

---

## 7. Next Single Action Recommendation

**Do not integrate `ChakrRelationalPointerMicro` into canonical production.**
Across Experiments 31, 32, and 33, we have rigorously established three scientific facts:
1. Pure autoregressive vocabulary softmax cannot copy out-of-vocabulary tokens ($0.0\%$).
2. Pointer attention enables out-of-vocabulary copying ($26\%-38\%$) and learns abstention ($F1 = 0.34$).
3. Single-head and two-stage linear pointer heads plateau at $\sim 28\%$ total retrieval accuracy under distractor pressure due to hidden state entanglement in the 3.44M parameter backbone.

**Single Action Recommendation:**
Conclude the pointer retrieval diagnostic series. Rather than engineering increasingly complex readout heads on a frozen 3.44M backbone, open **Milestone 34: Representation Binding Pretraining Diagnostic**, investigating whether multi-head relational binding objectives during pretraining or an expanded hidden dimension ($d_{model}=384$) are necessary for small transformers to maintain distractor-robust context representations.
