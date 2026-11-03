# ChakrView v0.9.0 — Model Card

## Model Details
- **Model Name:** ChakrView (ChakrMicro v0.1 Architecture)
- **Model Family:** Indigenous Decoder-Only Causal Transformer
- **Developer:** ChakrView Development Team (India)
- **Parameter Count:** 3,443,136 unique parameters (tied input/output embeddings: 786,432 weights shared)
- **Precision:** FP32 Native / CPU execution
- **Context Length:** 512 tokens
- **Positional Encoding:** Rotary Position Embeddings (RoPE, $\theta = 10000.0$)
- **Feedforward Layer:** SwiGLU ($d_{ffn} = 512$, exactly $\frac{8}{3} \times 192$)
- **Normalization:** Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Vocabulary Size:** 4,096 discrete BPE tokens (reserved IDs: 0 = `<BOS>`, 1 = `<EOS>`, 2 = `<PAD>`)
- **License:** MIT (Code) / Research Preview

---

## Intended Use
- **Primary Use:** Educational and academic research into low-resource language modeling, microscopic neural representation learning, and CPU-first deployment.
- **Secondary Use:** Benchmarking synthetic token extraction, entity copying, and algorithmic instruction tuning.
- **Out of Scope / Prohibited Use:** Clinical diagnostic recommendations, legal counsel, ungrounded autonomous decision making, financial advice, or critical safety systems.

---

## Capabilities and Empirical Limitations

### Observed Capabilities
1. **Low-Latency CPU Execution:** Under 30ms latency per 10 tokens on modern x86_64 CPUs without GPU requirements.
2. **Deterministic Token Continuation:** Fully deterministic greedy text continuation and reproducible sampling seeds.
3. **Structured Dialogue Formatting:** Encodes and generates within ChatML and standard dialogue delimiters without special token bleeding.

### Known Limitations
1. **Low General Factual Knowledge:** Because pretraining was performed on a small synthetic corpus (10,000 steps across 6.65M tokens), the model lacks broad world knowledge.
2. **Repetition and Attractor Dynamics:** Long-horizon greedy generation (>50 tokens) can enter attractor cycles or repetitive phrase loops if repetition penalties are disabled.
3. **Complex Reasoning:** Cannot perform multi-step arithmetic, logic, or ungrounded scientific reasoning.
