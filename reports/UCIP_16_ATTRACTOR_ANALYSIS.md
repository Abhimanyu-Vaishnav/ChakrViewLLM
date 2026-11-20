# UCIP-16 Attractor Dynamics & Autoregressive Degeneration Analysis

**Mission ID:** UCIP-16  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Auditor:** Antigravity Senior Forensic Auditor  
**Date:** 2026-11-20  

---

## 1. Identified Attractor Basins

Forensic analysis of 100 greedy generation traces identified two universal attractor basins:

### Attractor Basin A: English Cyclical Subword Loop
- **Surface Text:** `"The first of the first of the plant of the first..."`
- **Underlying Token Sequence:** `[898, 828, 117, 525, 770, 559, 828, 117, 525, 770, 559, 115, 636, 289, 770, 559, ...]`
  - 898: `'The '`
  - 828: `'fi'`
  - 117: `'r'`
  - 525: `'st '`
  - 770: `'of '`
  - 559: `'the '`
  - 115: `'p'`, 636: `'lan'`, 289: `'t '`
- **Occurrence Rate:** 58% of all greedy prompts (including English instruction, logic, QA, context, and arithmetic).

### Attractor Basin B: Wikipedia URL / Web Corpus Fragment Loop
- **Surface Text:** `"[https://www.wiv.gov/www.html Sublic Ishown..."`
- **Underlying Token Sequence:** `[94, 107, 119, 119, 115, 2393, 50, 50, 122, 122, 122, 49, 122, 108, 121, 49, ...]`
  - Single-character BPE tokens forming `http://www...`
- **Occurrence Rate:** 40% of all greedy prompts (primarily Hindi, Sanskrit, and multilingual code-switching).

---

## 2. Origin of Attractor Sequences

1. **Absence in SFT Dataset:**
   - Neither `"sublic"`, `"centron"`, `"wiv.gov"`, nor `"wign.com"` appears anywhere in `sft_train_12000.jsonl` or `sft_master_15000.jsonl`.
2. **Presence in Stage C Pretraining Shards:**
   - Shards in `data/tokenized/stage_c/train/` contain vast quantities of Wikipedia web dumps containing `http://`, `https://`, and standard encyclopedic citations.
3. **Autoregressive Reinforcement Mechanism:**
   - Because 75.5% of the gradient updates in UCIP-15 came from unmasked Stage C pretraining text, the model preserved high transition probabilities for common pretraining n-grams.
   - When faced with an out-of-distribution conversational prompt that it cannot resolve, the attention heads attend to generic frequent subwords (`"The "`, `'['`). Once emitted, the sequence falls into the deepest statistical basin of the pretraining distribution.

---

## 3. Logit and Probability Competition

At the first decoding position immediately following `Assistant: `:

| Candidate Token | Token String | Model Logit | Softmax Probability | Rank |
|---|---|---:|---:|:---:|
| **898** | `'The '` | **5.36** | **10.37%** | **1** |
| **38** | `'#'` | 4.91 | 6.64% | 2 |
| **76** | `'I'` | 4.21 | 3.29% | 3 |
| **94** | `'['` | 4.08 | 2.87% | 4 |
| **1** | `<EOS>` | **-0.73** | **0.02%** | **375** |

Because Token 898 (`"The "`) has a logit higher than all response tokens (and $6.09$ logits higher than `<EOS>`), greedy decoding almost invariably selects `'The '` or `'['`, immediately triggering Attractor Basin A or B.
