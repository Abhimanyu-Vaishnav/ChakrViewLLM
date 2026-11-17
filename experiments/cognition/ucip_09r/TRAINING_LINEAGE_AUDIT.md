# UCIP-09R: Training Lineage Forensic Audit

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Target Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
**Date:** November 17, 2026  

---

## 1. Lineage Verification Matrix

| Property | Value Recorded in Metadata | Verified Physical Artifact | Lineage Status |
|---|---|---|---|
| **Candidate Path** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | `experiments/cognition/ucip_09/checkpoints/candidate_best.pt` | **100% IDENTICAL BYTES** |
| **Candidate SHA-256** | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | **CONFIRMED** |
| **Parent Checkpoint** | `chakrview_ucip08_medium_model.pt` | `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt` | **CONFIRMED** |
| **Parent SHA-256** | `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631` | `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631` | **CONFIRMED** |
| **Training Script** | `experiments/cognition/ucip_09/train_ucip09.py` | Commit `75f208c` | **REPRODUCIBLE** |
| **Training Steps** | `step: 125` | Validated at step 125 | **CONFIRMED** |
| **Validation Loss** | `val_loss: 1.589271` | Log recorded `1.5893` | **100% MATCH** |
| **Training Data** | `experiments/cognition/ucip_09/data/ucip09_train.jsonl` (3,240 samples) | SHA-256: `2b251762...` | **CONFIRMED** |
| **Validation Data** | `experiments/cognition/ucip_09/data/ucip09_validation.jsonl` (360 samples) | SHA-256: `711ec499...` | **CONFIRMED** |
| **Tokenizer Artifact** | `data/experiments/vocab_4096` | 4,096 BPE (3,837 merges) | **CONFIRMED** |

---

## 2. Forensic Reconstruction Verdict

The release candidate `chakrview_ucip09_medium_model.pt` was **100% verified to have originated from the final corrected training run (`task-2473`)**, which implemented causal target shifting (`shift_logits = logits[:, :-1, :]`, `shift_targets = tgt[:, 1:]`) on parent checkpoint `chakrview_ucip08_medium_model.pt`.
