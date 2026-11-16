# ChakrView Conversational Model v0.10 (Release Candidate)

**Model Name:** ChakrView Conversational Model v0.10  
**Release Directory:** `releases/v0.10_conversational_model/`  
**Architecture:** ChakrMicro v0.1 (Decoder-Only Autoregressive Causal LM, RoPE, RMSNorm, SwiGLU)  
**Parameters:** 3,443,136  
**Context Window:** 512 tokens (Dynamic history packing up to 384 tokens, 48 token generation budget)  
**Tokenizer:** Chakr-BPE (4,096 vocabulary size, byte-backed fallback, 724 Devanagari merges)  
**Checkpoint File:** `chakrview_v010_model.pt`  
**Model SHA-256:** `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba`  

---

## Intended Use & Capabilities

- **Language Modes:** English, Devanagari Hindi, Roman Hindi, Hinglish.
- **Inference Requirements:** CPU-first execution (Intel Core i9 / i7 / i5 or compatible AMD64). Requires 0.5 GB RAM.
- **Interactive Chat Interface:**
  ```powershell
  python -m chakrview chat --checkpoint releases/v0.10_conversational_model/chakrview_v010_model.pt
  ```

---

## Known Scientific Limitations

1. **Multi-Subword Entity Binding:** In compact parameter regimes ($3.44\text{M}$ parameters), causal attention layers successfully output high-level conversational responses and Hindi/English vocabulary but fail to bind multi-token subword entities introduced earlier in the conversation (scoring $0.0\%$ on exact multi-turn name recall).
2. **Generative Priors vs. In-Context Memory:** The model prioritizes natural language fluency priors over long-range associative memory across turn boundaries.

