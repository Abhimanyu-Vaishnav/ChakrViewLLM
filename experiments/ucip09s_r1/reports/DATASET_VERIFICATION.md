# Stage 8 — Dataset and Token Claims Verification Report

**Date:** 2026-11-18  
**Audit Mission:** UCIP-09S-R1 Independent Evidence Verification & Evaluator Audit  
**Script:** `experiments/ucip09s_r1/scripts/verify_datasets_and_tokens.py`  
**Output Data:** `experiments/ucip09s_r1/evaluations/dataset_token_verification.json`

---

## 1. Verified Record Counts, Hashes, and Duplicates

All JSONL files were re-read and audited line by line. Dataset SHA-256 hashes, exact line counts, unique prompt counts, and exact duplicate counts are tabulated below:

| Dataset | Relative Path | SHA-256 Hash | Total Records | Unique Prompts | Duplicate Records |
|---|---|---|:---:|:---:|:---:|
| **UCIP-09 Train** | `experiments/cognition/ucip_09/data/ucip09_train.jsonl` | `2b251762e8216cfd956acdbd343e757e64bf2c33309cc24fa0f653e4ef37108d` | 3,240 | 3,229 | 0 |
| **UCIP-09 Val** | `experiments/cognition/ucip_09/data/ucip09_validation.jsonl` | `711ec499fef6073dfe1ad602adc7524596344334db835d3e7c477be77db48beb` | 360 | 360 | 0 |
| **UCIP-09R Train** | `experiments/cognition/ucip_09r/data/ucip09r_train.jsonl` | `d53f14e2fecd4f6370081a0f12c89792d8e7db47ec3b70fc9cff50f93101bcec` | 885 | 881 | 0 |
| **UCIP-09R Val** | `experiments/cognition/ucip_09r/data/ucip09r_val.jsonl` | `523c8b34e97a92b7dcd53b3fca788d0637f9c81db7b79f0753cd9c6b6036f5c2` | 98 | 98 | 0 |
| **UCIP-09S Bal Train** | `experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl` | `fc34eebd05aebb31c260281a0c9c0b0a84c32c714786b98536f37639c47eab68` | 1,080 | 1,079 | 0 |
| **UCIP-09S Bal Val** | `experiments/ucip09s/datasets/ucip09s_balanced_val.jsonl` | `707ed61b899face4471e710fe9d1a39f721dbf03c5f1d54a04aba912399aec71` | 120 | 120 | 0 |

---

## 2. Recomputed Category Distributions

| Category | UCIP-09 Train | Share | UCIP-09R Train | Share | UCIP-09S Bal Train | Share |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arithmetic / Math** | 549 | 16.9% | **783** | **88.5%** | 212 | 19.6% |
| **Exact Copying** | 355 | 11.0% | 50 | 5.6% | 109 | 10.1% |
| **English Conversation** | 448 | 13.8% | 12 | 1.4% | 215 | 19.9% |
| **Devanagari Hindi** | 545 | 16.8% | 12 | 1.4% | 221 | 20.5% |
| **English Explanations** | 449 | 13.9% | 22 | 2.5% | 88 | 8.1% |
| **Hinglish** | 635 | 19.6% | 6 | 0.7% | 90 | 8.3% |
| **Instruction Following** | 259 | 8.0% | **0** | **0.0%** | 145 | 13.4% |
| **Total** | 3,240 | 100.0% | 885 | 100.0% | 1,080 | 100.0% |

### Verification of Imbalance Claim:
The claim that UCIP-09R was **88.5% arithmetic (783 / 885)** is **100% verified**.
The claim that instruction following was **100% eliminated (0 samples)** in UCIP-09R is **100% verified**.
The claim that Hindi and English conversation were reduced to only 12 samples each (1.4%) is **100% verified**.

---

## 3. Numeric Word vs Tokenizer Token Frequencies

### Word-Level Frequencies in UCIP-09R Targets:
- Word `'21'`: **51 occurrences**
- Word `'27'`: **50 occurrences**
- Word `'15'`: **46 occurrences**
- Word `'18'`: **44 occurrences**
- Word `'24'`: **43 occurrences**
- Word `'30'`: **41 occurrences**
- Word `'33'`: **36 occurrences**
- Word `'12'`: **34 occurrences**
- Word `'22'`: **28 occurrences**
- Repetitive template phrases:
  - `"remaining."`: **178 occurrences** (20.1% of all samples!)
  - `"in total."`: **55 occurrences** (6.2% of all samples!)

### Tokenizer Token-Level Frequencies (Vocab 4096 BPE):
Target tokens encoded via `ChakrTokenizer` (`data/experiments/vocab_4096`):
- Token ID `127` (`' have'`): 237 occurrences
- Token ID `302` (`' will'`): 233 occurrences
- Token ID `186` (`' remaining'`): 178 occurrences
- Token ID `28` (`'.'`): 885 occurrences
- Token ID `512` (`' 21'`): 51 occurrences
- Token ID `548` (`' 27'`): 50 occurrences
- Token ID `438` (`' 15'`): 46 occurrences

### Strict Distinction between Text Statistics and Internal Representations:
- The presence of repetitive numbers and phrases in the training text is an **empirically proven textual and lexical fact**.
- However, prior claims that *"output projection probabilities were shifted"* or *"weight orthogonalization collapsed"* were **inferred deductions based on observable outputs (`21` answers) and loss gradients**, rather than direct internal layer probe activations. We clarify that internal weight vectors were not individually probed; rather, the behavioural collapse to `21` directly reflects the training data distribution.
