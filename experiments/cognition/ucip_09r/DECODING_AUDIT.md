# UCIP-09R: Decoding Audit & Factorial Control

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Controlled Decoding Results Across Arms

On the identical 100-probe reality benchmark, 5 distinct decoding configurations were evaluated:

| Decoding Arm | Repetition Penalty | No-Repeat N-gram | Pass Rate | Observed Lexical Behavior |
|---|---|---|---|---|
| **Arm A (Raw Greedy)** | 1.00 (Off) | 0 (Off) | 10.0% | High fluency on trained greetings; loops on explanations |
| **Arm B (RepPen 1.05)** | 1.05 | 0 (Off) | 11.0% | Soft penalty; stable grammar |
| **Arm C (NoRepeat 3 alone)** | 1.00 (Off) | 3 | **13.0%** | Breaks phrase looping without suppressing rare vocabulary |
| **Arm D (Chat CLI Default)** | 1.15 | 3 | 11.0% | Suppresses common syntax glue; forces tail tokens |
| **Arm E (Conservative)** | 1.05 | 3 | **13.0%** | Balanced; highest overall pass rate |

---

## 2. Selection of Default Configuration

For all subsequent evaluations and the interactive chat CLI:
- **Default Repetition Penalty:** `1.05` (reduced from 1.15)
- **Default No-Repeat N-gram:** `3`
- **Result:** Provides maximum defense against phrase looping while preventing lexical drift onto nonsensical tail words.
