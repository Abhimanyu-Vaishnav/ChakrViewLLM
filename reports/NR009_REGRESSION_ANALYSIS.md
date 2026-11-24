# CV-GENESIS-NR-009 Neural Regression Root Cause Analysis

## 1. Executive Summary

Mission **CV-GENESIS-NR-009 Phase 2** investigated why NR-008 reported 0% strict accuracy on its 14-task evaluation compared to 7.1% for NR-005.

Through controlled parameter-level, checkpoint-level, and token-level comparisons, **three conclusive technical root causes** were isolated:

---

## 2. Root Cause 1: Checkpoint Lineage Divergence
In NR-008 `train_sft.py`:
- `init_checkpoint` defaulted to `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` (the base pretraining checkpoint with no conversational instruction tuning).
- It was **not** initialized from `genesis_sft_best.pt`.
- Mean absolute weight difference between NR-008 weights and Canonical Pretrain is only `0.004803`, whereas the difference between NR-005 SFT Best and Canonical Pretrain is `0.013828`.
- **Consequence:** NR-008 was trained for only 3 epochs starting from raw pretrained weights rather than building cumulatively on the warm-started instruction-following foundation of NR-005.

---

## 3. Root Cause 2: Special Token Inversion (EOS / PAD Collision)
In NR-005:
- `eos_id = 1`
- `pad_id = 2`

In NR-008:
- `eos_id = 2`
- `pad_id = 0`

**Consequence:**
- In the canonical vocabulary (`data/experiments/vocab_4096/vocab.json`), Token ID 1 was supervised as `<eos>` across all Stage-C pretraining and NR-005 SFT.
- In NR-008, Token ID 2 was supervised as `<eos>` and padded with ID 0.
- When generating responses with `eos_token_id = 2`, the model encounters a severe distribution mismatch: tokens trained under ID 1 as sentence terminals are ignored, causing the model to generate past natural halting boundaries into garbled tail tokens.

---

## 4. Root Cause 3: Smoke Test Sample Flaw vs Statistical Reality
- The 14-prompt test was an informal smoke test with arbitrary string matching, not a statistically valid blind evaluation.
- When evaluated with `eos_token_id = 1` using the exact generation loop of NR-006, both NR-005 and NR-008 produce grammatical output, but the 5.18M capacity limitation prevents factual recall on zero-shot questions without retrieval.

---

## 5. Corrective Action Plan

1. **Phase 3:** Execute a full 300-prompt comparative blind benchmark across all 4 checkpoints (`nr005_best`, `nr005_final`, `nr008_best`, `nr008_final`) using the standardized NR-006 generation engine.
2. **Phase 4:** Re-align dataset special tokens (`eos_id=1, pad_id=2`) and warm-start future SFT directly from `genesis_sft_best.pt`.
3. **Phase 5:** Implement and pilot `ChakrGenesis-17M` to definitively break through the 5.18M capacity wall.
