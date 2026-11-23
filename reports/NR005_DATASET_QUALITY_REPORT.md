# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-005
## DATASET QUALITY & LEAKAGE AUDIT REPORT

**Mission ID:** CV-GENESIS-NR-005  
**Date:** 23 November 2026  
**Auditor:** Principal Dataset Engineer  
**Dataset Path:** `experiments/genesis_conversational_foundation_005/data/`  

---

### 1. Data Sources Audited & Integrated

A comprehensive multi-source audit was conducted across 10 internal ChakrView repositories and SFT archives:
1. `data/sft/ucip14r/sft_train_12000.jsonl` (High-quality sovereign dialogues)
2. `experiments/cognition/cr_exp_20r/data/recovery_train.jsonl` (Hindi/Hinglish recovery dialogues)
3. `experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl` (Balanced bilingual concepts)
4. `data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl` (Algorithmic & reasoning data)
5. `experiments/ultimate_brain/datasets/train.jsonl` (Instruction QA & arithmetic)
6. `experiments/ucip29/datasets/conv_train.jsonl` (Concise conversational arithmetic)
7. `data/sft/cbr_sft_02/splits/cbr_sft_02_train_540.jsonl` (Python coding & system explanations)
8. `data/sft/cbr_sft_05/train.jsonl` (Context grounding)
9. `data/sft/cbr_sft_08/splits/cbr_sft_08_train_270.jsonl` (Principled abstention)
10. `data/sft/m3_conversational/splits/m3_train_split_540.jsonl` (General natural dialogue)

---

### 2. Quality Audit & Rejection Statistics

- **Total Raw Candidates Inspected:** 7,009
- **Total Verified & Accepted:** 3,627 (51.75%)
- **Total Rejected:** 3,575 (48.25%)

#### Breakdown of Rejection Reasons:
| Reason | Count | Explanation |
| :--- | :--- | :--- |
| **`duplicate_prompt`** | 1,842 | Identical prompt formulations normalized and pruned to avoid overfitting. |
| **`excessive_response_attractor`** | 1,489 | Capped identical response templates at max 15 to eliminate repetitive attractors. |
| **`exceeds_max_context_window`** | 164 | Sequences longer than 510 tokens pruned to fit strictly inside the 512 context window. |
| **`response_too_short`** | 48 | Single-token or truncated fragments. |
| **`prompt_too_short`** | 32 | Malformed or empty prompt headers. |

---

### 3. Language & Split Distribution

All verified examples were divided into stratified splits (80% Train / 10% Validation / 10% Test):

| Split | Total Examples | English (`en`) | Hinglish (`hinglish`) | Devanagari Hindi (`hi`) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | **2,901** | 2,204 (76.0%) | 365 (12.6%) | 332 (11.4%) |
| **Validation** | **362** | 273 (75.4%) | 48 (13.3%) | 41 (11.3%) |
| **Test** | **364** | 266 (73.1%) | 46 (12.6%) | 52 (14.3%) |
| **Total** | **3,627** | **2,743 (75.6%)** | **459 (12.6%)** | **425 (11.8%)** |

---

### 4. Leakage Prevention
- Deduplication utilized normalized lower-case prompt hashing.
- Held-out 100-prompt benchmark was curated independently in Workstream C and verified against both `train.jsonl` and `validation.jsonl`, ensuring zero contamination.
