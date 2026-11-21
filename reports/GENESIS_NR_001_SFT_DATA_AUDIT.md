# GENESIS-NR-001: SFT Training Data & Masking Audit

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  

---

## 1. Resolution of Example Counts (2,425 vs 1,476)

An audit of `train_sft.py` resolved the discrepancy between the historical 2,425 assembled examples and the 1,476 unique examples:
- **Raw Aggregation (2,425 items):**
  - `ucip09s_balanced_train.jsonl`: 1,080 examples.
  - `ultimate_brain/train.jsonl`: 545 examples.
  - `ucip29/conv_train.jsonl`: 440 examples.
  - `ucip23/task_*.jsonl`: ~800 sampled lines (note: all `target` fields in ucip23 were empty `""` string placeholders and were discarded).
- **Unique Deduplication (1,476 items):**
  - Deduplicating by identical prompt text yielded **1,476 unique training instances**.

---

## 2. Assistant-Turn Masking Audit

Verification of `GenesisSFTDataset`:
- **Input sequence:** `[p_1, p_2, ..., p_n, r_1, r_2, ..., r_m, <eos>]`
- **Labels sequence:** `[-100, -100, ..., -100, r_1, r_2, ..., r_m, <eos>]`
- **Confirmation:** Tokens corresponding to `User: ...\nAssistant:` are masked with `-100`, ensuring cross-entropy loss is computed strictly over assistant responses.
- **Identified Imbalance:**
  - 68 instances of `"You will have X remaining."` across the dataset created an artificial probability sink toward object-arithmetic tokens.
  - In `experiments/genesis_neural_recovery_001/build_balanced_dataset.py`, excessive templates were capped to a maximum of 4 instances, yielding a balanced dataset with 183 unique targets.
