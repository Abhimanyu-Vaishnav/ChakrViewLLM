# ChakrView Technical Architecture Deep Dive

## 1. Executive Summary
ChakrView is an indigenous neural and cognitive architecture designed to combine a sovereign autoregressive neural core with modular cognitive subsystems.

## 2. Neural Foundation (`ChakrGenesis`)
- **Type:** Decoder-only Autoregressive Transformer
- **Parameters:** 5,177,600 (5.18M)
- **Layers:** 6 Transformer Blocks
- **Hidden Dim:** 256
- **Attention:** Grouped-Query Attention (GQA) with 8 Query heads and 2 Key-Value heads (head dim 32).
- **Positional Encoding:** Rotary Position Embedding (RoPE).
- **Feedforward:** SwiGLU with expansion dimension 682.
- **Normalization:** RMSNorm ($\epsilon=10^{-6}$).
- **Output:** Tied Embeddings (`vocab_size=4096`).

## 3. Cognitive Subsystems
1. **Working Memory Buffer:** Tracks active conversational entities and slots across multiple turns without fine-tuning weights.
2. **Span Pointer Network:** Extracts verbatim factual arguments and strings from reference documents.
3. **Deterministic Tool Interface:** Routes exact arithmetic, date calculation, and code AST verification to deterministic local solvers.
4. **Principled Abstention Gate:** Detects unanswerable or future-predictive queries and enforces safe abstention.
