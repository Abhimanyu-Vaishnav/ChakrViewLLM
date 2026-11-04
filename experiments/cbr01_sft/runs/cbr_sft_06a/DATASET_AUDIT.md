# CBR-SFT-06A Dataset Audit & Integrity Report

**Date:** 2026-11-04  
**Audit Status:** PASSED / ZERO CONTAMINATION  
**Auditor:** Antigravity Autonomous Research Agent  

---

## 1. Compositional Architecture

The CBR-SFT-06A dataset addresses the catastrophic forgetting and mode collapse identified in CBR-SFT-05 by introducing an explicitly balanced, leak-free tripartite structure:

| Task Group | Description | Total Count | Train Split (90%) | Val Split (10%) | Effective Share |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Group A: Replay** | Audited conversational greetings, identity & boundaries from CBR-04 | 80 | 72 | 8 | **47.06%** |
| **Group B: Copy Grounding** | Strictly verified single-token digits, numbers & keywords | 70 | 63 | 7 | **41.18%** |
| **Group C: Diverse Refusal**| 5 distinct natural phrasing variants on genuinely absent context | 20 | 18 | 2 | **11.76%** |
| **Total** | | **170** | **153** | **17** | **100.0%** |

---

## 2. Sampling Clarification

- **Dataset Total:** 170 examples (153 train, 17 validation).
- **Proportion Notice:** Group A constitutes **47.06%** of the training split ($72 / 153$).
- **Effective Exposure:** Under uniform random batch sampling with an effective batch size of 4, each batch encounters approximately ~1.88 replay examples, ~1.65 copy grounding examples, and ~0.47 abstention examples.
- **Breaking the Verbatim Attractor:** In contrast to CBR-SFT-05 (where a single verbatim string was repeated 20 times), Group C uses **5 diverse refusal variants**, ensuring that no single phrase accounts for more than $2.3\%$ of the total dataset.

---

## 3. Cryptographic Hashes & Leakage Audit

### File Hashes:
- `data/sft/cbr_sft_06a/dataset.jsonl`: `59a81624cc51204d912537f1be327837f8bf2bafffe46d4d1384eef58f90c49c`
- `data/sft/cbr_sft_06a/splits/cbr_sft_06a_train_153.jsonl`: `bfed72c23b1e4504a560340aed4e6d4845c633c4f7326b603459d6a79ae10ccd`
- `data/sft/cbr_sft_06a/splits/cbr_sft_06a_val_17.jsonl`: `11c276839fc30a37b631c545debbad371217f21b3fb20e8a32e08602d907247f`

### Leakage Audit Results:
- Cross-split contamination between `train_153` and `val_17`: **0 matching prompts (PASS)**
- Prompt leakage against `held_out_cbr_sft_04_80.json`: **0 matching prompts (PASS)**
- Prompt leakage against `held_out_cbr_sft_05_120.json`: **0 matching prompts (PASS)**
- Prompt leakage against `held_out_greeting_identity_70.json`: **0 matching prompts (PASS)**
- All single-token copy targets in Group B were confirmed to tokenize to exactly 1 token under `vocab_4096`.

