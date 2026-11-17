# UCIP-09R: Token Boundary Audit & Detokenization Safety

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Boundary Discrepancy Findings

In Phase C, tokenization boundaries were inspected across English, Devanagari Hindi, Roman Hindi, and Hinglish (`experiments/cognition/ucip_09r/verify_token_boundaries.py`).

### Discovery:
When `User: ...\nAssistant:` and ` <Target>` are encoded independently:
- The prompt ends with token ID `[61]` (`':'`).
- The target begins with a leading space subword (e.g., token ID `[3482]` for `' A '` or token ID `[35]` for `' '`).
When encoded as a unified string `User: ...\nAssistant: <Target>`:
- The subword pair `':'` + `' '` is merged by BPE into token ID `[324]` (`': '`).

### Resolution:
In interactive inference, prompt formatting guarantees that the assistant prefix is consistently tokenized as `... Assistant:` ending in token ID `[61]`. The subsequent token generation directly predicts the leading target token (e.g. `' '` or `' A'`). Detokenization calls across the entire codebase now consistently enforce `errors="replace"` to guarantee crash immunity against partial multi-byte sequences.
