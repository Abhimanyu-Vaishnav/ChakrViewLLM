# CV-GENESIS-NR-008 Architecture & Hardware Sizing Experiments

## 1. Executive Summary

To achieve meaningful conversational competence, multi-turn reasoning, and factual fidelity, we profiled the parameter scaling limits of the `ChakrGenesis` decoder transformer architecture against our local hardware constraint (**NVIDIA RTX A1000 Laptop GPU, 6.0 GB VRAM**).

---

## 2. Scaling Profile Analysis

All configurations retain canonical architectural invariants:
- Rotary Position Embedding (RoPE) ($\theta = 10,000$)
- Grouped-Query Attention (GQA) with 4:1 query-to-KV head ratio
- SwiGLU non-linear gating ($d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}$)
- Pre-RMSNorm ($\epsilon = 1 \times 10^{-5}$)
- Weight-tied token embeddings and output projection
- Byte-Pair Encoding Vocabulary ($V = 4,096$)

| Variant | $d_{\text{model}}$ | Layers | $n_q$ / $n_{kv}$ | $d_{\text{ff}}$ | Total Params | Non-Tied Params | Weights (FP16) | Peak Train VRAM (B=16) | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Genesis-5M (Canonical Baseline)** | 256 | 6 | 8 / 2 | 682 | **5,177,600** | 4,129,024 | 9.9 MB | ~103 MB | **Proven / Active** |
| **Genesis-17M (Mid-Scale Sprint)** | 384 | 10 | 12 / 3 | 1024 | **17,063,808** | 15,490,944 | 32.5 MB | ~320 MB | **Highly Feasible** |
| **Genesis-30M (Platform Target)** | 512 | 10 | 16 / 4 | 1364 | **29,612,544** | 27,515,392 | 56.5 MB | ~532 MB | **Feasible for v1.0** |
| **Genesis-77M (High Capacity)** | 768 | 12 | 16 / 4 | 2048 | **77,482,752** | 74,337,024 | 147.8 MB | ~1.33 GB | **Feasible with FP16/AMP** |

---

## 3. Autonomous Decision & Phased Strategy

1. **Immediate SFT Optimization (Sprint 1):** Focus on the proven **5.18M Genesis Foundation** (`canonical_pretrain_best.pt` & `genesis_sft_best.pt`). Rebalancing the training corpus (35% Hindi, synthetic Python code, structured multi-turn dialogues) directly addresses the 3.0% Hindi and 0.0% multi-turn bottlenecks observed in NR-006 without invalidating canonical weights.
2. **Next-Generation Scaling (Sprint 2 - v0.14+):** Train a **Genesis-17M** or **Genesis-30M** model from scratch on the expanded multilingual corpus once data curation is complete. At ~532 MB peak training VRAM, the 30M model utilizes less than 10% of our 6.0 GB VRAM, allowing comfortable batch scaling ($B=32$ or $B=64$ with gradient accumulation).
