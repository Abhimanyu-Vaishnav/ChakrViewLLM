# ChakrView Exp04 — Preflight Findings & Architecture Audit

**Target Directory**: `experiments/research_v09_conversation_sft/exp04_pointer_robustness/`  
**Date**: 2026-10-22  
**Purpose**: Document preflight audit findings, confirmed root causes of label corruption, exact checkpoint and artifact paths, model contracts, and benchmark design specifications before executing the robustness challenge.

---

## 1. Verified Cryptographic Hashes & Protected Files

| Artifact | Repository Path | Expected SHA-256 | Verified Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **BIT-IDENTICAL** |
| **Canonical Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119...` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **BIT-IDENTICAL** |
| **Canonical Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dc...` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **BIT-IDENTICAL** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51...` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNTOUCHED** |
| **Exp15 Pointer Checkpoint** | `experiments/research_v09_conversation_sft/exp03_field_conditioned_pointer/checkpoints/pointer_best.pt` | Recorded | `8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d` | **VERIFIED** |

---

## 2. Root Causes of Label Corruption in Prior Pointer Benchmark

Inspection of `generate_pointer_data.py` in Exp15 identified three critical issues that contaminated the Exp14/Exp15 pointer evaluation:
1. **Sublist Token Mismatch on Trailing Whitespace**:
   - The canonical BPE tokenizer frequently merges a trailing space into the final token of an entity string (e.g. `'0 '` instead of `'0'`).
   - `locate_target_span` performed naive literal integer sublist comparison (`tokens[i:i+L] == target_tokens`). When target was tokenized in isolation without context, the token sequence differed.
2. **Silent Fallback to Abstention `[0, 0]`**:
   - When the sublist match failed, `generate_pointer_data.py` fell back to `start_pos, end_pos = 0, 0` rather than raising an error.
   - This corrupted 21 training examples, 10 validation examples, and 14 test examples—labeling them as "abstention" despite the entity being present in the prompt.
3. **Template Positional Regularity**:
   - The synthetic templates positioned entities at near-constant token offsets (e.g., function names always at tokens 13–18). The pointer heads learned to associate `field_id` with absolute position indices.

---

## 3. Verified Field-ID Contract & Model Specifications

The evaluated checkpoint (`pointer_best.pt`) expects:
- `num_fields = 10`, corresponding to:
  - `0`: `dynamic_entity_binding`
  - `1`: `turn_dependent_retrieval`
  - `2`: `anaphora_reference_resolution`
  - `3`: `superseded_instruction_adherence`
  - `4`: `user_correction_handling`
  - `5`: `distractor_resistance`
  - `6`: `ambiguity_clarification`
  - `7`: `missing_context_abstention`
  - `8`: `factual_uncertainty_handling`
  - `9`: `constrained_code_synthesis`
- **Model Interface**:
  - `forward(input_ids, field_ids, candidate_mask, use_field_conditioning=True)`
  - `predict_span(input_ids, field_id, candidate_mask, use_field_conditioning=True)` returns `(best_s, best_e, extracted_tokens, confidence)`.
  - Position `[0, 0]` represents explicit abstention (`[NO_ANSWER]`).
  - Constrained decoding enforces $1 \le s \le e \le s + 12$ or $(s=0, e=0)$.

---

## 4. Fail-Closed Alignment Strategy for Exp04

Exp04 implements a byte-level token alignment adapter:
1. Every token in `input_ids` is mapped to its exact byte range $[s_{\text{byte}}, e_{\text{byte}}]$ within the UTF-8 encoded prompt bytes.
2. The target substring's character span $[c_{\text{start}}, c_{\text{end}}]$ in the prompt is converted to UTF-8 byte span $[b_{\text{start}}, b_{\text{end}}]$.
3. The token span $[t_{\text{start}}, t_{\text{end}}]$ is calculated from all tokens that overlap with $[b_{\text{start}}, b_{\text{end}}]$.
4. The decoded token span is checked: `tokenizer.decode(input_ids[t_start:t_end+1]).strip() == target_text.strip()`.
5. If normalization or alignment fails, or if a non-abstention target cannot be cleanly resolved, the alignment pipeline raises an explicit `AlignmentError`. Silent reassignment to `[0, 0]` is strictly forbidden.
