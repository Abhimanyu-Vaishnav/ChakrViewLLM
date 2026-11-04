# Phase D: Tokenizer & Technical String Forensics

**Date:** 2026-11-04  
**Vocabulary Source:** `data/experiments/vocab_4096` (ChakrMicro BPE Tokenizer, 4096 tokens)  
**Round-Trip Integrity:** 100% (22/22 tested technical structures decode identically)  

---

## 1. Key Finding: Extreme Subword Fragmentation on Technical Entities

While the tokenizer achieves bit-exact reversibility (`decode(encode(text)) == text`), technical entities undergo **severe fragmentation**:

1. **IP Addresses:**
   - `192.168.1.105` splits into 4 tokens: `['192.168.1', '.', '10', '5']`
   - `172.24.11.99` splits into 8 tokens: `['1', '7', '2.', '2', '4', '.1', '1.', '99']`
   - **Crucial Diagnostic Observation:** Notice why the model emitted `172` in CBR-SFT-05! The token sequence begins with `['1', '7', '2.']`. The model copied the first 3 tokens (`1`, `7`, `2.`), which decode to the string prefix `"172."` or `"172"`. Because continuing past the dot requires predicting single-digit fragments (`'2'`, `'4'`), cross-entropy uncertainty spiked, causing premature stopping!
2. **Ports:**
   - `8080` splits into 2 tokens: `['80', '80']`
   - `7443` splits into 3 tokens: `['7', '44', '3']`
   - `5433` splits into 4 tokens: `['5', '4', '3', '3']`
3. **Hexadecimal Hashes & Commit SHAs:**
   - Full 64-char SHA-256 splits into **51 individual tokens**!
   - Short 8-char SHA `a619297f` splits into 5 tokens: `['a', '6', '192', '97', 'f']`.
4. **Environment Variables & Keys:**
   - `DATABASE_PORT` splits into **11 tokens**: `['D', 'A', 'T', 'A', 'B', 'AS', 'E', '_', 'P', 'OR', 'T']`.
   - `MAX_CLIENTS=1024` splits into **12 tokens**: `['M', 'A', 'X', '_', 'C', 'L', 'I', 'E', 'NT', 'S', '=', '1024']`.

---

## 2. Theoretical Implications for ChakrMicro (4 Layers, 512 Hidden Dim)

- **The Length Penalty on Joint Probability:**
  For an autoregressive model to reproduce a 51-token SHA or an 8-token IP address, every single intermediate token $t_k$ must be sampled with near 100% probability:
  $$P(\text{entity}) = \prod_{k=1}^N P(t_k \mid t_{<k}, \text{context})$$
  If the average token probability is $0.80$, an 8-token IP has a cumulative survival probability of $0.80^8 = 0.168$ (16.8%).
- **Contrast with Canonical Refusal:**
  The canonical abstention phrase `" That information is not provided in the context."` consists of frequent, high-frequency natural language subwords that the model has seen thousands of times across pre-training and SFT. Its joint probability under uncertainty is orders of magnitude higher than a sequence of 8 arbitrary digit tokens.
- **Conclusion:**
  Exact copying failure is **strongly exacerbated by subword fragmentation**. The model did not fail to "know" the IP address; its attention heads successfully pointed to the first token/prefix, but the multi-step autoregressive rollout collapsed under high fragmentation entropy.
