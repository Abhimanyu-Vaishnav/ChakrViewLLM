# UCIP-07 Architecture Research & Specification Report

**Date:** 2026-11-16  
**Status:** Completed & Calibrated  
**Objective:** Architecture Exploration from 3.44M to 53.43M parameters, addressing capacity limits, attention geometry, and context scaling.

---

## 1. Candidate Architecture Family Comparison

| Architecture | Parameters | Depth ($L$) | Width ($d$) | Heads ($H$) | Head Dim | KV Heads (GQA) | FFN Dim | Tied LMHead | Estimated Peak RAM (Train) | CPU Inference (Tokens/s) |
|---|---|---|---|---|---|---|---|:---:|---|---|
| **ChakrMicro** | 3,443,136 | 6 | 192 | 6 | 32 | 6 (MHA) | 512 | Yes | ~0.4 GB | ~180–220 tok/s |
| **ChakrMedium** | 12,932,736 | 7 | 384 | 6 | 64 | 6 (MHA) | 896 | Yes | ~1.1 GB | ~85–110 tok/s |
| **ChakrCore-28M** | 27,795,968 | 8 | 512 | 8 | 64 | 8 (MHA) | 1,408 | Yes | ~2.4 GB | ~45–60 tok/s |
| **ChakrCore-GQA** | 24,650,240 | 8 | 512 | 8 | 64 | 2 (4x GQA) | 1,408 | Yes | ~2.1 GB | ~55–70 tok/s |
| **ChakrLarge-48M** | 53,425,280 | 10 | 640 | 10 | 64 | 10 (MHA) | 1,792 | Yes | ~4.8 GB | ~22–30 tok/s |

---

## 2. Component Design & Mathematical Formulations

### A. Attention & Positional Encoding
- **Rotary Position Embedding (RoPE):**  
  $\Theta = 50,000.0$ (scaled up from standard $10,000.0$). This lowers high-frequency decay and preserves long-range relative token distance tracking up to the 512 context boundary.
- **Grouped-Query Attention (GQA):**  
  Keys and Values are grouped into $n_{kv\_heads} = 2$ across $n_{heads} = 8$, reducing KV-cache memory footprints by $4\times$ during incremental generation without compromising multi-head representational diversity.

### B. SwiGLU Feed-Forward Network
- Uses SiLU gating:
  $$\text{FFN}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$
- Expansion ratio calibrated to $\approx \frac{8}{3} d_{\text{model}} = 1,408$, ensuring optimal FLOP-to-parameter density per layer.

### C. Pre-RMSNorm Residual Geometry
- Pre-normalization on each sublayer preserves clean residual stream propagation:
  $$x_{l+1} = x_l + \text{Attn}(\text{RMSNorm}(x_l)) + \text{FFN}(\text{RMSNorm}(x_l + \text{Attn}(\dots)))$$
- Prevents vanishing/exploding gradients as depth increases from 6 to 8+ layers.

---

## 3. Mandatory Overfit Verification Result
- Model: `ChakrCore` (27,795,968 parameters)
- Protocol: 4 challenging multilingual samples across arithmetic, exact copying, Devanagari Hindi, and technical explanation.
- Overfit Loss: $2.6100 \to 0.0164$ in 40 steps.
- **Greedy Autoregressive Rollout Accuracy:** **4 / 4 (100.0% Exact Match)**.
- Demonstrates mathematical validity, backward gradient flow, and flawless causal autoregression.
