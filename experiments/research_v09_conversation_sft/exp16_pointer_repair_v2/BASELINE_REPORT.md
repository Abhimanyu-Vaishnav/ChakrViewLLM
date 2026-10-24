# ChakrView Exp16 — Read-Only Baseline & Defect Characterization

**Baseline Date**: 2026-10-22  
**Target Repair Directory**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/`  
**Audited Reference Directory**: `experiments/research_v09_conversation_sft/exp16_pointer_retraining_design/`  

---

## 1. Verified Cryptographic Hashes of Baseline Artifacts

All baseline artifacts, canonical checkpoints, vocabulary, merges, protected runtime code, and Exp15 models were verified:

| Artifact | Repository Path | Initial SHA-256 Hash |
| :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Canonical Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` |
| **Exp15 Pointer Model** | `experiments/.../exp03_field_conditioned_pointer/.../pointer_best.pt` | `8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d` |
| **Original Exp16 Train** | `experiments/.../exp16_pointer_retraining_design/data/train_exp16.jsonl` | `55abc82154801da753f07bf1cf0ab1226ac5f07fc2506550d2c69b02d3372804` |
| **Original Exp16 Val** | `experiments/.../exp16_pointer_retraining_design/data/val_exp16.jsonl` | `569a0091223aca34a84606a99ac60416b1efe3b0fdddf4d72b174b1b854c59a7` |
| **Original Exp16 Test** | `experiments/.../exp16_pointer_retraining_design/data/test_exp16.jsonl` | `48de3477de8e50ba605165c093f2e329f16edf734ecdd4899b2c02e18e4885ad` |

---

## 2. Quantitative Baseline Distribution Inventory

- **Split Sizes**: Train = 300, Val = 60, Test = 120 (Total = 480).
- **Abstention Counts / Rates**:
  - Train: 132 / 300 (44.00%)
  - Val: 24 / 60 (40.00%)
  - Test: 50 / 120 (41.67%)
- **Context Token Lengths**:
  - Train: min 65, max 209, mean 129.4 tokens.
  - Val: min 74, max 212, mean 131.6 tokens.
  - Test: min 65, max 207, mean 131.1 tokens.
- **Answerable Target Span Token Lengths**:
  - min 2 tokens, max 13 tokens.

---

## 3. Confirmed Baseline Defects in Original Exp16 Dataset

1. **Exact Cross-Split Prompt Duplication (6 overlaps)**:
   - `EXP16_TRAIN_TURN_RET_001` vs `EXP16_TEST_TURN_RET_001` (identical frequency dialogue and target `412 MHz`).
   - `EXP16_TRAIN_TURN_RET_003` vs `EXP16_VAL_TURN_RET_003` vs `EXP16_TEST_TURN_RET_003`.
   - `EXP16_TRAIN_TURN_RET_028` vs `EXP16_TEST_TURN_RET_011`.
   - Shared template seeds produced exact prompt duplicates across splits.
2. **Category-Level Deterministic Shortcuts**:
   - `dynamic_entity_binding`: 100% of answerable targets are candidate 1 (`sensor_*_1`).
   - `distractor_resistance`: 100% of answerable targets are `'OPERATIONAL'`.
   - `anaphora_reference_resolution`: 100% of answerable targets are `'Cluster_*_Alpha_*'`.
   - `ambiguity_clarification`: 100% of answerable targets are `'pod_*_cyan_*'`.
   - `user_correction_handling`: 100% of answerable targets are `'worker_*_new_*'`.
3. **Model Contract Violation (`max_span_len = 12`)**:
   - 9 examples in train/val (`constrained_code_synthesis`, e.g. `verify_digest_va_0`) produce spans of length 13 tokens.
   - The decoding constraint `for e in range(s, min(s + self.max_span_len, T))` strictly prevents extracting spans $> 12$ tokens, making these examples unreachable during evaluation.

---

## 4. Architectural Requirements for Repaired Pipeline (`v2`)

1. **Scenario-Family Decoupling**: Each split must be generated from disjoint scenario domains (e.g. Train = Industrial Power & Robotics; Val = Telecom & Satellite; Test = Marine & Aerospace) with separate sentence templates and distinct candidate entity pools.
2. **Dynamic Slot Randomization**: Uniformly randomize target selection among distractors (e.g. randomly choose candidate 0, 1, or 2; balance Alpha vs Beta; balance target status values across the pool).
3. **Span Length Normalization**: Shorten synthetic code identifiers (e.g. `verify_hash_*`, `calc_crc_*`, `run_audit_*`) to ensure token lengths remain between 5 and 9 tokens, safely under the `max_span_len = 12` ceiling.
