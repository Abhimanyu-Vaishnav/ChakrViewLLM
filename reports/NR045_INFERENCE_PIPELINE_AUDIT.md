# NR-045: Inference Pipeline Audit Report

## 1. Executive Summary

This report documents the end-to-end trace of a user prompt through the complete ChakrView 1.0 runtime (`chakrview/app/brain_runtime.py`) and isolates the divergence points between pure model generation and runtime generation.

---

## 2. End-to-End Inference Trace Analysis

```
User Input ("Hello!")
      │
      ▼
1. Operational Self-Model Check ("who are you", etc.) -> Intercepts identity queries
      │
      ▼
2. Memory Query ("remember", etc.) -> Intercepts stored facts
      │
      ▼
3. Neuro-Symbolic Tool Route (Arithmetic/String/List keywords) -> Intercepts tools
      │
      ▼
4. Pure Neural Path:
   a. Prompt Formatting: `<user>\n{turn_history}\n{user_input}\n<assistant>\n`
   b. Tokenization: `tokenizer.encode(...)`
   c. RoPE Positional Embedding: `RotaryEmbedding.forward()`
   d. Forward Pass: `ChakrGenesis17M.forward()`
   e. Sampling: Argmax (T=0) or Multinomial (T>0)
   f. Token Decoding: `tokenizer.decode(..., errors="replace")`
   g. Cleanup & Post-processing: Strips trailing `<user>` tags.
```

---

## 3. Discovered Inference Pipeline Discrepancies

1. **Prompt Template Desynchronization:**
   - As established in Dataset Forensics, `brain_runtime.py` formats input as:
     `<user>\n{user_input}\n<assistant>\n`
   - However, the underlying RC1 weights were trained predominantly on:
     `<user>\nUser: {user_input}\nAssistant: \n<assistant>\n`
   - This single formatting gap caused immediate out-of-distribution confusion at token position 0 of the generated response.
2. **RoPE Maximum Context Handling:**
   - In `chakr_genesis_17m.py`, RoPE static pre-computation cached only up to `max_seq_len = 256` in the saved config.
   - When context history exceeded 256 tokens during multi-turn chat, a tensor dimension mismatch occurred (`size 257 vs 256`).
   - Dynamic caching was added to resolve the crash, but training the model on short 256-token contexts limits multi-turn context span.
3. **Absence of Repetition Penalty:**
   - The runtime generated tokens purely via argmax or raw multinomial sampling without a repetition penalty.
   - When encountering low-confidence token predictions, the model fell into short cyclic loops (e.g., repeating `<eos>\n12<eos>\n24 + 12 = 120.<eos>`).
