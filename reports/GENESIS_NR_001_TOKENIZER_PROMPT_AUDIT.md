# GENESIS-NR-001: Tokenizer and Prompt Pipeline Audit

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  
**Tokenizer:** `experiments/tokenizer/candidates/v4096` (`vocab_size=4096`)  

---

## 1. Encode / Decode Lossless Verification

Tested the 6 required audit strings using the frozen BPE tokenizer with full UTF-8 byte fallback:

| Input Text | Token Count | Encoded Token IDs | Decoded Text | Lossless Status |
| :--- | :---: | :--- | :--- | :---: |
| `Hello, how are you?` | 11 | `[987, 111, 314, 273, 573, 673, 535, 124, 114, 120, 66]` | `Hello, how are you?` | **PASSED (Lossless)** |
| `Namaste, tum kaise ho?` | 10 | `[3895, 426, 582, 273, 472, 357, 632, 1483, 573, 66]` | `Namaste, tum kaise ho?` | **PASSED (Lossless)** |
| `I am building a small AI model.` | 10 | `[988, 1000, 1976, 630, 301, 2375, 1802, 1300, 874, 49]` | `I am building a small AI model.` | **PASSED (Lossless)** |
| `Explain how a computer stores information.` | 15 | `[3840, 115, 1133, 350, 573, 673, 301, 799, 390, 2584, 601, 267, 1238, 406, 49]` | `Explain how a computer stores information.` | **PASSED (Lossless)** |
| `The capital of Australia is Canberra.` | 18 | `[894, 1320, 382, 324, 766, 68, 120, 477, 286, 108, 301, 589, 70, 284, 101, 281, 439, 49]` | `The capital of Australia is Canberra.` | **PASSED (Lossless)** |
| `I have five boxes with eight books each.` | 18 | `[988, 312, 595, 826, 595, 1001, 123, 356, 627, 104, 614, 287, 1001, 114, 2338, 104, 524, 49]` | `I have five boxes with eight books each.` | **PASSED (Lossless)** |

---

## 2. Special Token & Prompt Formatting Alignment

- **Special Tokens:**
  - `<pad>`: ID `0`
  - `<bos>`: ID `1`
  - `<eos>`: ID `2`
- **Prompt Structure:**
  - Standardized format: `User: <input>\nAssistant:`
  - Note: Trailing spaces after `Assistant:` (e.g. `Assistant: `) shift the final token from `':'` (token 61) to `': '` (token 324), altering position logits. Prompt assembly in `agent_controller.py` was normalized to ensure strict consistency with SFT training prompts.
