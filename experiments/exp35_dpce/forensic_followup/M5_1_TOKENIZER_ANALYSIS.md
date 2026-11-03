# CHAKRVIEW — M5.1 TOKENIZER INVESTIGATION
## Empirical Analysis of Subword Fragmentation and Vocabulary Representation in ChakrMicro

**Tokenizer:** Canonical BPE (4,096 Vocabulary, SHA-256 `db4d6119...`)  
**Experiment:** Exp35-DPCE Forensic Audit  
**Date:** 2026-10-30  

---

## 1. SUBWORD FRAGMENTATION OF ENTITIES

An analysis of entities used in the Exp35 dataset and benchmark demonstrated severe subword fragmentation:

| Entity Category | Entity String | Token Count | Subword Token IDs | Subword String Fragments |
|---|---|---|---|---|
| **Names (Training)** | `Aarav` | 3 | `[68, 280, 2306]` | `['A', 'ar', 'av']` |
| | `Meera` | 3 | `[2260, 283, 100]` | `['Me', 'er', 'a']` |
| | `Kabir` | 4 | `[78, 412, 108, 117]` | `['K', 'ab', 'i', 'r']` |
| | `Nisha` | 3 | `[81, 388, 314]` | `['N', 'is', 'ha']` |
| | `Ananya` | 4 | `[68, 113, 286, 2423]`| `['A', 'n', 'an', 'ya']` |
| **Names (Unseen Benchmark)**| `Zoya` | 3 | `[93, 114, 2423]` | `['Z', 'o', 'ya']` |
| | `Tariq` | 4 | `[87, 280, 108, 116]` | `['T', 'ar', 'i', 'q']` |
| | `Bhavna` | 4 | `[69, 314, 121, 709]` | `['B', 'ha', 'v', 'na']` |
| **Cities** | `Rivapur` | 5 | `[85, 108, 121, 494, 326]` | `['R', 'i', 'v', 'ap', 'ur']` |
| | `Chandrapur` | 6 | `[70, 107, 286, 103, 385, 2382]` | `['C', 'h', 'an', 'd', 'ra', 'pur']` |
| | `Kashipur` | 4 | `[78, 427, 453, 2382]` | `['K', 'as', 'hi', 'pur']` |
| **Codes** | `KX-104` | 5 | `[78, 91, 48, 545, 55]` | `['K', 'X', '-', '10', '4']` |
| | `SERIAL_NUM` | 9 | `[86, 2244, 76, 68, 79, 98, 81, 88, 80]` | `['S', 'ER', 'I', 'A', 'L', '_', 'N', 'U', 'M']` |

---

## 2. THE MECHANICAL COPYING IMPLICATION

1. **Multi-Step Autoregressive Copying:**  
   Because almost no entity exists as a single token, a model cannot copy an entity with a single attention lookup. To copy `Chandrapur`, the network must:
   - Step 1: Attend to prompt position of `C`, output token `70`.
   - Step 2: Attend to prompt position of `h`, output token `107`.
   - Step 3: Attend to `an`, output token `286`.
   - Step 4: Attend to `d`, output token `103`.
   - Step 5: Attend to `ra`, output token `385`.
   - Step 6: Attend to `pur`, output token `2382`.
2. **Induction Circuit Complexity:**  
   In mechanistic interpretability, an induction head requires tracking the previous token $S_{i-1}$ and looking up the prompt occurrence of $S_{i-1}$ to emit $S_i$. When an entity is fragmented into 4–6 subwords, this chain must be maintained error-free across 6 successive sampling steps.
3. **Verdict on Tokenizer:**  
   While subword fragmentation does not make copying mathematically impossible, it **massively compounds the difficulty** for a 3.44M parameter model ($d_{\text{model}}=192$), making un-pretrained induction heads virtually incapable of completing multi-token entity sequences.

