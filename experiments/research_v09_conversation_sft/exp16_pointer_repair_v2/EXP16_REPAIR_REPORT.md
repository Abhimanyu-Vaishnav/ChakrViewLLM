# ChakrView Exp16 (v2) — Repaired Dataset Engineering Report

**Status**: `READY FOR HUMAN SEMANTIC REVIEW — NO BLOCKING TECHNICAL ISSUES FOUND`  
**Repair Target Directory**: [`experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/)  
**Date**: 2026-10-22  

---

## 1. Executive Summary

This report documents the resolution of the three blocking defects identified during the semantic audit of the original Exp16 dataset generation pipeline:

1. **Defect 1 (Cross-Split Prompt Duplication)**:  
   *Root Cause*: Shared filler pools and identical index counters produced 6 exact prompt duplicates between Train, Validation, and Test.  
   *Resolution*: Implemented complete **scenario-family domain decoupling**:
   - **Train**: Industrial Power Grid & Robotics (`train_exp16_v2.jsonl`, seed `1111`, frequency offset $+1000$).
   - **Val**: Telecom Cellular & Cloud Datacenters (`val_exp16_v2.jsonl`, seed `2222`, frequency offset $+3000$).
   - **Test**: Aerospace Avionics & Autonomous Marine (`test_exp16_v2.jsonl`, seed `3333`, frequency offset $+5000$).
   - *Result*: **Zero prompt duplicates** cross splits ($0 / 480$).

2. **Defect 2 (Deterministic Category-Level Shortcuts)**:  
   *Root Cause*: Hardcoded slot indexing meant 100% of answerable dynamic binding targets were candidate 1 (`sensor_*_1`), 100% of distractor targets were `'OPERATIONAL'`, 100% of anaphora targets were `'Alpha'`, and 100% of ambiguity targets were `'cyan'`.  
   *Resolution*: Applied **dynamic slot randomization**:
   - `dynamic_entity_binding`: Uniformly balanced across candidate slots 0, 1, and 2 ($8:8:8$ in Train).
   - `distractor_resistance`: Balanced across 5 operational states (`OPERATIONAL`, `ONLINE`, `ACTIVE`, `READY`, `RUNNING`).
   - `anaphora_reference_resolution`: Balanced between `Alpha` and `Beta` ($50/50$).
   - `ambiguity_clarification`: Balanced across 4 distinct colors per split.
   - `user_correction_handling`: Target destination nodes randomized across diverse functional roles (`node`, `host`, `core`, `edge`, `gw`, `srv`).

3. **Defect 3 (Span-Length Contract Violation)**:  
   *Root Cause*: Function names like `verify_digest_va_0` produced 13 tokens, exceeding `max_span_len = 12`.  
   *Resolution*: Compacted synthetic identifiers (`calc_crc_*`, `run_audit_*`, `verify_hash_*`) and node role names.  
   *Result*: All 282 answerable spans across train, val, and test are strictly $\le 12$ tokens (range: 2 to 12 tokens).

---

## 2. Before vs After Dataset Distribution Comparison

| Dimension | Original Exp16 (v1) | Repaired Exp16 (v2) | Status |
| :--- | :---: | :---: | :---: |
| **Train Examples** | 300 | 300 | Preserved |
| **Val Examples** | 60 | 60 | Preserved |
| **Test Examples** | 120 | 120 | Preserved |
| **Total Examples** | 480 | 480 | Preserved |
| **Categories per Split** | 10 balanced | 10 balanced | Preserved |
| **Train Abstentions** | 132 (44.0%) | 126 (42.0%) | Verified |
| **Val Abstentions** | 24 (40.0%) | 24 (40.0%) | Verified |
| **Test Abstentions** | 50 (41.7%) | 48 (40.0%) | Verified |
| **Cross-Split Prompt Overlaps** | **6 duplicates** | **0 duplicates** | **RESOLVED** |
| **Dynamic Binding Slot Bias** | Slot 1: 100% | Slots 0, 1, 2 balanced | **RESOLVED** |
| **Distractor State Bias** | 'OPERATIONAL': 100% | 5 states balanced | **RESOLVED** |
| **Anaphora Referent Bias** | 'Alpha': 100% | 'Alpha' & 'Beta' 50/50 | **RESOLVED** |
| **Spans Exceeding max_span_len (12)** | **9 violations** | **0 violations** | **RESOLVED** |

---

## 3. Cryptographic Hashes & Preservation Audit

All historical files and protected artifacts remain completely untouched:

| Artifact | Repository Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Unmodified |
| **Canonical Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Unmodified |
| **Canonical Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Unmodified |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | Unmodified |
| **Exp15 Pointer Model** | `experiments/.../exp03_field_conditioned_pointer/.../pointer_best.pt` | `8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d` | Unmodified |
| **Orig Exp16 Train** | `experiments/.../exp16_pointer_retraining_design/data/train_exp16.jsonl` | `55abc82154801da753f07bf1cf0ab1226ac5f07fc2506550d2c69b02d3372804` | Unmodified |
| **Orig Exp16 Val** | `experiments/.../exp16_pointer_retraining_design/data/val_exp16.jsonl` | `569a0091223aca34a84606a99ac60416b1efe3b0fdddf4d72b174b1b854c59a7` | Unmodified |
| **Orig Exp16 Test** | `experiments/.../exp16_pointer_retraining_design/data/test_exp16.jsonl` | `48de3477de8e50ba605165c093f2e329f16edf734ecdd4899b2c02e18e4885ad` | Unmodified |
| **Repaired Train (v2)**| `experiments/.../exp16_pointer_repair_v2/data/train_exp16_v2.jsonl` | `8dbb7952a65825272a8c3e86c0788647dc5fe4272ceb0ae796c5aaefeaae0b49` | Created |
| **Repaired Val (v2)** | `experiments/.../exp16_pointer_repair_v2/data/val_exp16_v2.jsonl` | `a3f5a01bc2a98f126da351509930f40ce7ef6e9628dd30c30a84e314646731ba` | Created |
| **Repaired Test (v2)**| `experiments/.../exp16_pointer_repair_v2/data/test_exp16_v2.jsonl` | `9b36203cfd347895e6945a8df24d32a9bf8696ec694a973d09a061b4745ae32b` | Created |

---

## 4. Newly Created Files in Exp16 Repair Directory

All work is self-contained in `experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/`:
1. `BASELINE_REPORT.md`: Baseline defect inventory and hash audit.
2. `generate_repaired_datasets.py`: Deterministic v2 dataset generation pipeline.
3. `data/train_exp16_v2.jsonl`: 300 repaired training examples.
4. `data/val_exp16_v2.jsonl`: 60 repaired validation examples.
5. `data/test_exp16_v2.jsonl`: 120 repaired held-out test examples.
6. `EXP16_REPAIR_MANIFEST.json`: Configuration manifest and verified hashes.
7. `HUMAN_REVIEW_QUEUE.md`: Complete review queue for all 480 examples marked `[ ] AWAITING HUMAN REVIEW`.
8. `test_exp16_repaired.py`: 8 automated regression and validation unit tests.
9. `EXP16_REPAIR_TEST_REPORT.md`: Automated test execution report.
10. `EXP16_REPAIR_REPORT.md`: Comprehensive engineering report.

---

## 5. Baselines for Subsequent Evaluation

When authorized to evaluate, the repaired dataset establishes the following baseline expectations:
- **Always-Abstain Baseline**: Predicts $[0, 0]$ unconditionally. On test ($N=120$, 48 abstentions), this baseline achieves **40.00% exact-match accuracy** ($48 / 120$). Any trained pointer must significantly exceed $40.00\%$ to demonstrate learned capability.
- **First-Match Heuristic Baseline**: A lexical scan for the first candidate entity achieves $\approx 30–35\%$ accuracy due to distractor saturation.
- **Pre-Registered Success Threshold**: $\ge \mathbf{75.00\%}$ exact-span accuracy across the complete 120-example test set ($90 / 120$).

---

## 6. What Remains Unverified & Next Action

- **Human Semantic Review**: While technical alignment, decoding, and span lengths are 100% verified by automated tests, the 480 examples in [`HUMAN_REVIEW_QUEUE.md`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp16_pointer_repair_v2/HUMAN_REVIEW_QUEUE.md) await human review of semantic edge cases.
- **Single Safest Next Action**: Wait for explicit user authorization. Do not train, evaluate, or perform Git operations.
