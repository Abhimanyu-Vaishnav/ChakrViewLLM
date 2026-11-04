# Phase F: Abstention Attractor Investigation & Mode Collapse

**Date:** 2026-11-04  
**Audit Target:** `data/sft/cbr_sft_05/dataset.jsonl` (400 examples)  

---

## 1. Discovery: The Single Dominant Text Mode in the Entire Dataset

An exhaustive frequency analysis of target responses in the CBR-SFT-05 dataset reveals an overwhelming statistical imbalance:

| Rank | Target Response String | Occurrences in Dataset | Percentage of Total Dataset |
| :---: | :--- | :---: | :---: |
| **#1** | **`'That information is not provided in the context.'`** | **20** | **5.0%** |
| #2 | `'32'` | 4 | 1.0% |
| #3 | `'Delhi'` | 3 | 0.75% |
| #4 | `'ChakrMicro'` | 3 | 0.75% |
| #5 | `'ap-south-1'` | 3 | 0.75% |
| #6+ | All unique arbitrary strings (IPs, hashes, ports, paths) | 1 each | 0.25% each |

---

## 2. Why the Model Collapsed into Abstention

1. **Information-Theoretic Attractor State:**
   - Every entity target in the exact copy, port, IP, and hash categories is **unique** (1 occurrence in 400 examples).
   - The abstention string appears **20 times**, with identical token transitions across 10 consecutive tokens.
   - For an optimizer (AdamW, LR $1.2 \times 10^{-4}$), predicting the abstention phrase yields a predictable, guaranteed loss reduction on every single abstention batch.
   - By contrast, predicting arbitrary 8-token IPs (`172.24.11.99`) incurs high entropy penalties if any single token diverges.
2. **The Shortcut Mechanism:**
   - When queried with `"What is the value of X?"`, the network learned that emitting the sequence starting with `That information...` minimizes average cross-entropy across all queries whenever confidence on the specific entity is below a certain threshold.
   - This directly explains why by Step 200, **every single key-value query emitted `'That information is not provided in the context.'`**, causing 100% accuracy on abstention tests and 0% accuracy on key-value extraction.
3. **Remediation Strategy for CBR-SFT-06:**
   - **Do not repeat identical verbatim abstention strings.** Use a pool of diverse, short, natural abstentions (`"Not specified."`, `"The context does not state this."`, `"Unknown from provided text."`).
   - Limit abstention to $\le 3\%$ of total batches during initial copy induction, or train exact copy and epistemic refusal in separate curricula.
