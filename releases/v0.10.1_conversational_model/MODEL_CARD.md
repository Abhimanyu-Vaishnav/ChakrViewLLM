# ChakrView Conversational Model v0.10.1 (Experimental Capability Recovery Release)

**Model Version:** v0.10.1  
**Release Directory:** `releases/v0.10.1_conversational_model/`  
**Checkpoint File:** `chakrview_v0101_model.pt`  
**SHA-256 Hash:** `548b20a73d3cb3a662bcbfe74ab4f108d7bfdc4c60e56abd76049b260fa16a3b`  
**Model Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, decoder-only causal transformer, RoPE, RMSNorm, SwiGLU)  
**Context Window:** 512 tokens (384 max prompt tokens with dynamic turn packing, 48 generation tokens)  
**Vocabulary:** Chakr-BPE (4,096 tokens, byte-backed fallback, 724 Devanagari subword merges)  

---

## Capabilities & Evaluation Summary

1. **Multilingual Script Alignment:**
   - **English:** Full persona retention (`"I am ChakrView, an indigenous neural language model developed from first principles."`) and greeting fluency ($100\%$ completion rate).
   - **Devanagari Hindi:** $95.0\%$ script appropriateness. Generates authentic Devanagari tokens and broken free from single-phrase attractor loops.
   - **Roman Hindi & Hinglish:** $100\%$ conversational relevance, integrating colloquial vocabulary (`naam`, `bhai`, `kya`, `hai`, `kar rahe ho`).
2. **Execution & Hardware:**
   - $100\%$ local CPU inference via `python -m chakrview chat --checkpoint releases/v0.10.1_conversational_model/chakrview_v0101_model.pt`.
   - Requires $<0.5\text{GB}$ RAM.

---

## Known Scientific Limitations

- **Multi-Subword Entity Binding ($0.0\%$ Exact Recall):**
  At 3.44M parameters, the self-attention layers successfully generate conversational frames (`"Your name is..."` / `"Tumhara naam..."`) but cannot parametrically bind multi-subword entities introduced across long intervening turns. Factual memory requires either scaled model capacity ($>50\text{M}$ parameters) or external state indexing.

