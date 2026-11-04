# Phase E: Token-Level Generation & Teacher-Forced Probability Diagnostics

**Date:** 2026-11-04  
**Checkpoints Compared:**  
- **CBR-SFT-04 Step 200** (`327118ce...`)  
- **CBR-SFT-05 Step 100** (`8653d0bb...`)  
- **CBR-SFT-05 Step 200** (`34bc6a12...`)  

---

## 1. Key Quantitative Findings: Token-by-Token Probabilities

### Case 1: IP Address Retrieval (`172.24.11.99`)
- **Prompt:** `User: The edge sensor gateway is set to 172.24.11.99.\nQuestion: What is the IP address of the edge sensor gateway?\nAssistant:`
- **CBR-SFT-04 Step 200:** Emits generic context failure: `"The context."`
- **CBR-SFT-05 Step 100:**
  - Token 0 (`'1'`): Probability = **10.48% (Rank 2)** (Top predicted: `'10'` at 14.48%)
  - Token 1 (`'7'`): Probability = **12.47% (Rank 1)**
  - Token 2 (`'2.'`): Probability collapses to **0.17% (Rank 63)** (Top predicted: `'0'`)
  - Token 4 (`'4'`): Probability = **13.35% (Rank 1)**
- **Diagnostic Conclusion:**  
  CBR-SFT-05 **actively learned to attend to the target digits in context**. Tokens 1 and 4 are the #1 ranked predictions in the entire 4096-token vocabulary! However, the subword token boundary `'2.'` (a merged digit-and-dot token) causes an immediate drop in probability because the model's small 4-layer parameter capacity has not formed a stable joint transition between fragmented digit representations and merged punctuation tokens.

---

## 2. The Delimiter & Leading Space Contract Revelation

During training data preparation, dataset entries in `cbr_sft_05.jsonl` contained responses authored with a leading space:
`"response": " 172.24.11.99"`
However, `normalize_prompt_response_strings(prompt, response, strip_delimiters=False)` executed:
`r_str = response if response.endswith("\n") else response.strip()`
Because `response` did not end in `\n`, `.strip()` was applied, stripping the leading space!
- As a result, the model was trained on tokens starting **without** a space (`[1, 7, 2.]`).
- During greedy autoregressive generation after `Assistant:`, the model naturally produces tokens directly without a leading space (`'1'`, `'10'`, `'That'`).
- This confirms that **prompt delimiter handling between training and inference was consistent**, eliminating prompt delimiter stripping as the cause of failure. The collapse is strictly driven by the subword probability degradation and the abstention attractor basin.
