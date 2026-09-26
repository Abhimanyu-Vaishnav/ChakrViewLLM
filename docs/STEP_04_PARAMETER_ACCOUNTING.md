# ChakrView Step 4: Parameter Accounting Report

**Document Version**: 1.0.0  
**Phase**: Step 4 — Parameter Accounting  
**Model Target**: Chakr-Micro v0.1  
**Status**: VERIFIED & RECONCILED  

---

## 1. Executive Summary

This report establishes the exact parameter accounting for the **Chakr-Micro v0.1** neural core. Every parameter formula is derived analytically from architectural dimensions ($V = 4096, d_{\text{model}} = 192, N = 6, H = 6, d_{\text{head}} = 32, d_{\text{ff}} = 512, b = 0$) and reconciled programmatically against the actual model implementation.

---

## 2. Mathematical Formula vs Programmatic Count

| Subcomponent | Analytical Formula | Mathematical Expected | Code-Generated Count | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Token Embedding ($E$)** | $V \times d_{\text{model}} = 4096 \times 192$ | $786,432$ | $786,432$ | **MATCH** |
| **Attention $W_Q$ (per layer)** | $d_{\text{model}} \times d_{\text{model}} = 192 \times 192$ | $36,864$ | $36,864$ | **MATCH** |
| **Attention $W_K$ (per layer)** | $d_{\text{model}} \times d_{\text{model}} = 192 \times 192$ | $36,864$ | $36,864$ | **MATCH** |
| **Attention $W_V$ (per layer)** | $d_{\text{model}} \times d_{\text{model}} = 192 \times 192$ | $36,864$ | $36,864$ | **MATCH** |
| **Attention $W_O$ (per layer)** | $d_{\text{model}} \times d_{\text{model}} = 192 \times 192$ | $36,864$ | $36,864$ | **MATCH** |
| **Attention Sub-layer Total (per layer)** | $4 \times 36,864$ | **$147,456$** | **$147,456$** | **MATCH** |
| **Pre-Attention RMSNorm 1 (per layer)** | $d_{\text{model}} = 192$ | $192$ | $192$ | **MATCH** |
| **SwiGLU $W_{\text{gate}}$ (per layer)** | $d_{\text{model}} \times d_{\text{ff}} = 192 \times 512$ | $98,304$ | $98,304$ | **MATCH** |
| **SwiGLU $W_{\text{up}}$ (per layer)** | $d_{\text{model}} \times d_{\text{ff}} = 192 \times 512$ | $98,304$ | $98,304$ | **MATCH** |
| **SwiGLU $W_{\text{down}}$ (per layer)** | $d_{\text{ff}} \times d_{\text{model}} = 512 \times 192$ | $98,304$ | $98,304$ | **MATCH** |
| **SwiGLU FFN Total (per layer)** | $3 \times 98,304$ | **$294,912$** | **$294,912$** | **MATCH** |
| **Pre-FFN RMSNorm 2 (per layer)** | $d_{\text{model}} = 192$ | $192$ | $192$ | **MATCH** |
| **Single Transformer Block Total** | $147,456 + 192 + 294,912 + 192$ | **$442,752$** | **$442,752$** | **MATCH** |
| **All $N = 6$ Transformer Blocks** | $6 \times 442,752$ | **$2,656,512$** | **$2,656,512$** | **MATCH** |
| **Final RMSNorm** | $d_{\text{model}} = 192$ | $192$ | $192$ | **MATCH** |
| **Tied LM Output Head ($W_{\text{out}}$)** | Shared with $E$ ($0$ additional unique parameters) | $0$ | $0$ | **MATCH** |
| **TOTAL UNIQUE PARAMETERS** | $786,432 + 2,656,512 + 192$ | **$3,443,136$** | **$3,443,136$** | **MATCH** |
| **TOTAL TRAINABLE PARAMETERS** | All unique parameters requiring grad | **$3,443,136$** | **$3,443,136$** | **MATCH** |

---

## 3. Parameter Distribution & Proportions

```
Total Parameters: 3,443,136 (100.00%)
├── Token Embedding (E):         786,432  (22.84%)
├── Transformer Blocks (6x):   2,656,512  (77.15%)
│   ├── Attention Sub-layers:    884,736  (25.69%)  [6 x 147,456]
│   ├── SwiGLU Sub-layers:     1,769,472  (51.39%)  [6 x 294,912]
│   └── Layer Normalizations:      2,304   (0.07%)  [6 x (192 + 192)]
├── Final RMSNorm:                   192   (0.01%)
└── Tied LM Head (W_out):              0   (0.00%)  [Shares memory with E]
```

---

## 4. Weight Tying Verification

Weight tying is verified at the hardware pointer level:
```python
assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
```
Because the LM Head reuses the transposed input embedding tensor $\mathbf{E}^T$, an additional $4096 \times 192 = 786,432$ parameters ($3.14\text{ MB}$ at FP32) are eliminated from static model memory.

---

## 5. Precision & Static Weight Footprint

| Data Type | Bytes / Parameter | Exact Memory (Bytes) | Memory (MiB) | Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **FP32** | 4 | $13,772,544$ | $13.13\text{ MiB}$ | $13.77\text{ MB}$ |
| **FP16 / BF16** | 2 | $6,886,272$ | $6.57\text{ MiB}$ | $6.89\text{ MB}$ |
| **INT8** | 1 | $3,443,136$ | $3.28\text{ MiB}$ | $3.44\text{ MB}$ |
| **INT4** | 0.5 | $1,721,568$ | $1.64\text{ MiB}$ | $1.72\text{ MB}$ |

---
*End of Parameter Accounting Report*
