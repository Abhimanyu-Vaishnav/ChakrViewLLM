# ChakrView Exp04 — Evaluation Protocol

**Date**: 2026-10-22  
**Target Suite**: Independent Pointer Robustness Challenge v1.0  
**Benchmark SHA-256**: `11c0b4bfc33410e113c9c112f35e50c8a9b4446ff93fb930f0f837b0ee9d9748`  
**Dataset Path**: `experiments/research_v09_conversation_sft/exp04_pointer_robustness/data/exp04_challenge.jsonl`

---

## 1. Frozen Model and Tokenizer Artifacts

The evaluation tests the existing trained Exp15 pointer without fine-tuning, training, or weight mutation:
- **Trained Pointer Checkpoint**: `experiments/research_v09_conversation_sft/exp03_field_conditioned_pointer/checkpoints/pointer_best.pt`
  - SHA-256: `8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d`
- **Canonical Backbone Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Canonical Vocab**: `data/experiments/vocab_4096/vocab.json`
  - SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- **Canonical Merges**: `data/experiments/vocab_4096/merges.json`
  - SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`

---

## 2. Field-ID Contract & Predefined Mappings

| Category | Field ID | Evaluation Slot Description |
| :--- | :---: | :--- |
| `dynamic_entity_binding` | 0 | Active runtime entity name |
| `turn_dependent_retrieval` | 1 | Most recent numerical value / state |
| `anaphora_reference_resolution` | 2 | Resolved entity or abstention on ambiguity |
| `superseded_instruction_adherence` | 3 | Latest override serialization format |
| `user_correction_handling` | 4 | Explicitly corrected target entity |
| `distractor_resistance` | 5 | Targeted device state ignoring distractors |
| `ambiguity_clarification` | 6 | Disambiguated pod or abstention |
| `missing_context_abstention` | 7 | Non-contextual query (must abstain [0,0]) |
| `factual_uncertainty_handling` | 8 | Inherently speculative query (must abstain [0,0]) |
| `constrained_code_synthesis` | 9 | Approved function identifier |

---

## 3. Evaluation Rules & Exact-Match Criteria

1. **Inference Input**: `input_ids` (tokenized prompt prepended with BOS=0), `field_id`, and `candidate_mask`. Gold answers, labels, and spans are strictly forbidden at inference time.
2. **Decoding Constraints**: Maximum span length = 12 tokens ($s \le e \le s + 12$) or $(s=0, e=0)$.
3. **Exact-Span Match**: A prediction is correct if and only if both start and end token indices match the gold span ($s_{\text{pred}} == s_{\text{gold}}$ and $e_{\text{pred}} == e_{\text{gold}}$).
4. **Abstention Definition**: Position `[0, 0]` represents explicit abstention. An example is correct if gold is `[0, 0]` and predicted is `[0, 0]`.
5. **Pre-Registered Acceptance Threshold**:
   - $\ge \mathbf{75.0\%}$ exact-span accuracy across the complete 120-example challenge suite ($90 / 120$).
   - If performance falls below 75.0%, status is `ROBUSTNESS THRESHOLD NOT MET`.
