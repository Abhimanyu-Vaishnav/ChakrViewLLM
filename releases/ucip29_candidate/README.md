# ChakrView Conversation Model v0.1-alpha (ChakrConv)

**Version:** 0.1-alpha  
**Architecture:** ChakrConv (6 layers, 6 heads, d_model=192, hidden_dim=512, SwiGLU, RoPE)  
**Tied Embeddings:** True (Vocab size = 4,096)  
**Total Parameters:** 3,443,136  
**Primary Checkpoint:** `chakrconv_v01_best.pt`  
**Checkpoint SHA-256:** `0f55f25a1c202f456186c3c0c99575b823df6b4c8f85cb61ebac60aaa9b8fa05`  
**Primary Execution Backend:** CPU-first (PyTorch 2.x)  
**Inference Speed:** ~149.2 tokens/second on Intel CPU  
**First-token Latency:** 65–125 ms  

---

## 1. Model Overview

ChakrConv v0.1-alpha is an indigenous, decoder-only conversational language model built from scratch as part of **UCIP-29**. Designed specifically to resolve the generation boundary collapse identified in UCIP-26, UCIP-27, and UCIP-28, it introduces:
- Scale-balanced skip connections preventing late-layer FFN prior dominance.
- Bias-free SwiGLU activation networks initialized with residual damping ($1/\sqrt{2 \cdot N_{\text{layers}}}$).
- Zero external pretrained weights, zero hosted LLM API distillation, and zero proprietary tokenizer dependencies.

---

## 2. Release Artifacts

| Artifact | File Path | SHA-256 Digest |
|---|---|---|
| **Checkpoint** | `releases/ucip29_candidate/chakrconv_v01_best.pt` | `0f55f25a1c202f456186c3c0c99575b823df6b4c8f85cb61ebac60aaa9b8fa05` |
| **Model Code** | `releases/ucip29_candidate/chakrconv.py` | — |
| **Chat CLI** | `releases/ucip29_candidate/chat.py` | — |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc0a5d2906b3a28b03e05a8f4c653f29b46df7c0bc78bb51d93` |
| **Tokenizer Merges**| `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962071869e5d996bb6b5c3ffc4b7bdf0175b2a09c258d4a6639` |

---

## 3. How to Run Locally

### Interactive Chat Session
From the project root:

```powershell
.\.venv\Scripts\python.exe releases\ucip29_candidate\chat.py
```

Inside the chat interface:
- Type messages directly to converse with the neural model.
- `/reset` : Clears conversational history and begins a fresh session.
- `/info` : Displays active parameter count, architecture, and checkpoint digest.
- `/exit` : Exits the CLI cleanly.

---

## 4. Benchmark Performance & Known Limitations

### Quantitative Evaluation (120 Sealed Held-Out Test Records)
- **Exact Match Accuracy:** 1.67%
- **Attractor Elimination:**
  - Token `'#'` (Step-800 Pretraining Attractor): **0.00%** (0 / 120) — down from 100% in Step-800.
  - Token `'A'` (UCIP-20 SFT Attractor): **0.00%** (0 / 120) — down from 34.2% in UCIP-20.
  - Infinite Repetitive Loops: **0.00%** (0 / 120) — down from 9.2% in UCIP-20.

### Known Limitations
1. **Limited Context Length Capacity:** The small 3.4M parameter capacity restricts long multi-turn semantic reasoning; after 2–3 turns, responses tend to degenerate into generic phrases.
2. **Arithmetic Reliability:** Simple additions are partially recognized, but multi-digit arithmetic or multiplications frequently predict approximate tokens.
3. **Classification:** This candidate is officially classified as a **RESEARCH PREVIEW READY — LIMITATIONS REMAIN**. It is genuinely runnable locally on CPU, executes real neural autoregressive generation without templates or shortcuts, but requires scale expansion for production deployment.
