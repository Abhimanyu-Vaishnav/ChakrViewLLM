# COGNITIVE SCALING ROADMAP: CPU-FIRST (CR-001)

This roadmap outlines the progressive scaling trajectory for ChakrView from ChakrMicro v0.1 up to edge-deployable cognitive cores, prioritizing consumer CPU execution, zero-copy sharding, and memory efficiency.

---

## 1. Multi-Tier Architecture Progression

| Architecture Tier | Layers | Width ($d_{model}$) | Attention Heads | FFN Dimension | Parameters | Training RAM (FP32 AdamW) | Inference Latency (CPU) | Cognitive Target |
|---|---|---|---|---|---|---|---|---|
| **Tier 0: ChakrMicro v0.1** (Frozen Baseline) | 6 | 192 | 6 | 512 | **3,443,136** | ~39.4 MB | <0.8 ms / token (>1,200 tok/s) | Syntax, Greetings, Basic Key-Value Copying, 1-digit Math |
| **Tier 1: ChakrSmall v0.2** (Induction Specialist)| 8 | 288 | 6 | 768 | **9,147,168** | ~104.7 MB | ~1.8 ms / token (~550 tok/s) | Dedicated 2-layer induction circuits, 2-digit arithmetic, syllogisms |
| **Tier 2: ChakrMedium v0.2** (Reasoning Core) | 10 | 384 | 6 | 1024 | **19,275,648**| ~220.6 MB | ~3.8 ms / token (~260 tok/s) | Multi-hop problem solving, constraint planning, bilingual logic |
| **Tier 3: ChakrLarge v0.2** (Edge Foundation) | 12 | 512 | 8 | 1536 | **43,004,416**| ~492.2 MB | ~8.5 ms / token (~120 tok/s) | Full compositional reasoning, code generation, metacognition |

---

## 2. Progressive Phasing Strategy

```
Phase A: Baseline Curriculum Consolidation (ChakrMicro v0.1 - 3.44M)
- Train on Levels 1-3 (Semantic relations, variable binding, simple arithmetic) on Stage C corpus.
- Establish proof-of-concept that small transformers can learn variable binding with clean data.

Phase B: Isolated Induction Scaling (ChakrSmall v0.2 - 9.15M)
- Implement ChakrSmall in an isolated experimental branch (`exp/chakr-small-01`).
- Train on 25M-50M tokens of balanced multilingual and synthetic logic data.
- Benchmark against Tier 0 on the 12-domain Cognitive Evaluation Suite.

Phase C: Architectural Enhancements (ChakrMedium v0.2 - 19.3M)
- Introduce auxiliary validity heads and scratchpad tokens.
- Evaluate multi-step GSM-Chakr reasoning and self-correction.

Phase D: Quantized Edge Deployment (GGUF / INT4 / INT8)
- Quantize ChakrMedium to INT4 (~11 MB footprint) for zero-dependency execution on low-cost devices (Raspberry Pi, mobile ARM, consumer laptops).

