# CR-EXP-20: Phase D — Architecture Decision & Model Profile

**Mission ID:** CR-EXP-20  
**Phase:** Phase D — Architecture Development & Profiling  
**Date:** 2026-11-15  
**Configuration File:** `experiments/cognition/cr_exp_20/configs/cr20_model_config.json`  

---

## 1. Executive Architecture Decision

Following the hardware profiling (Intel Core i9 13th Gen, 32GB RAM, PyTorch CPU-first) and Phase B tokenizer findings:
1. **Model Framework:** `ChakrMicro` decoder-only autoregressive causal transformer with rotary position embeddings and RMSNorm / SwiGLU activation blocks.
2. **Parameters & Dimensions:**
   - Vocabulary Size: 4,096 tokens (exact match to Chakr-BPE).
   - $d_{\text{model}}$: 192.
   - Layers: 6.
   - Attention Heads: 6 ($d_{\text{head}} = 32$).
   - Feed-Forward Dimension: 512.
   - Maximum Sequence Length: 512 tokens.
   - Total Parameters: **3,443,136 parameters**.
3. **Weight Transfer & Initialization Strategy:**
   - By retaining the 4,096 vocabulary and $3.44\text{M}$ parameter topology, we safely transfer weights from the verified conversational parent checkpoint `cbr_sft_04` (SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`).
   - This provides immediate conversational warm-start (English and Hinglish conversational semantics) while allowing the model to acquire Devanagari Hindi, Roman Hindi, and multi-turn context retention through our balanced curriculum.

---

## 2. Forward / Backward Pass Verification

- Forward pass input: `[B=2, T=128]`
- Forward pass output: `[B=2, T=128, V=4096]`
- Backward pass & gradient finiteness: Verified. No NaN/Inf gradients.
- Context window ceiling: 512 tokens (384 tokens reserved for prompt, 48 tokens for generation).

