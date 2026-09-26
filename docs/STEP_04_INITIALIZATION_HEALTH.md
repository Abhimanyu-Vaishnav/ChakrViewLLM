# ChakrView — Step 4.1: Neural Core Initialization Health

## Executive Summary

A complete numerical inspection was conducted on the initialized weights of **ChakrMicro v0.1**.
- **Total Trainable Parameters**: 3,443,136
- **NaN Count / Percentage**: 0 (0.00%)
- **Inf Count / Percentage**: 0 (0.00%)
- **Accidental All-Zero Tensors**: None
- **Status**: **PASS — HEALTHY AND FINITE**

---

## 1. Parameter Group Summary

| Parameter Group | Param Count | Mean | Std Dev | Min | Max | % Zeros | % NaN | % Inf |
|---|---|---|---|---|---|---|---|---|
| **Embedding** | 786,432 | -0.000051 | 0.020029 | -0.092358 | +0.090686 | 0.0000% | 0.00% | 0.00% |
| **Attention Q/K/V Projections** | 663,552 | -0.000005 | 0.019972 | -0.096667 | +0.102101 | 0.0000% | 0.00% | 0.00% |
| **Attention Out Projections (Residual)** | 221,184 | +0.000027 | 0.005771 | -0.026207 | +0.025259 | 0.0000% | 0.00% | 0.00% |
| **FFN Gate/Up Projections** | 1,179,648 | +0.000003 | 0.019997 | -0.095344 | +0.096196 | 0.0000% | 0.00% | 0.00% |
| **FFN Down Projections (Residual)** | 589,824 | +0.000010 | 0.005773 | -0.025529 | +0.026090 | 0.0000% | 0.00% | 0.00% |
| **RMSNorm Scales (Layer & Final)** | 2,496 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% | 0.00% | 0.00% |

### Key Statistical Observations:
1. **Standard Projections & Embeddings**: Standard deviation matches target `initializer_range = 0.0200`.
2. **Residual Projections (Out & Down)**: Scaled by $1 / \sqrt{2N} = 1 / \sqrt{12} \approx 0.2887$, target $\sigma \approx 0.005774$. Measured $\sigma \approx 0.005773$, matching theoretical variance reduction to stabilize deep residual addition.
3. **RMSNorm Scale Vectors**: Initialized exactly to constant $1.000000$ (identity transformation at $t=0$), with zero variance.

---

## 2. Granular Tensor Inspection

| Parameter Name | Tensor Shape | Elements | Mean | Std Dev | Min | Max | % Zeros |
|---|---|---|---|---|---|---|---|
| `embedding.weight` | `[4096, 192]` | 786,432 | -0.000051 | 0.020029 | -0.092358 | +0.090686 | 0.0000% |
| `layers.0.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.0.attn.q_proj.weight` | `[192, 192]` | 36,864 | -0.000230 | 0.020054 | -0.079660 | +0.085637 | 0.0000% |
| `layers.0.attn.k_proj.weight` | `[192, 192]` | 36,864 | +0.000308 | 0.019765 | -0.074152 | +0.087527 | 0.0000% |
| `layers.0.attn.v_proj.weight` | `[192, 192]` | 36,864 | -0.000052 | 0.020029 | -0.084142 | +0.076801 | 0.0000% |
| `layers.0.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000051 | 0.005773 | -0.022740 | +0.022721 | 0.0000% |
| `layers.0.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.0.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | +0.000019 | 0.020030 | -0.081570 | +0.083637 | 0.0000% |
| `layers.0.ffn.up_proj.weight` | `[512, 192]` | 98,304 | -0.000022 | 0.020026 | -0.083326 | +0.082799 | 0.0000% |
| `layers.0.ffn.down_proj.weight` | `[192, 512]` | 98,304 | -0.000004 | 0.005764 | -0.024447 | +0.025870 | 0.0000% |
| `layers.1.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.1.attn.q_proj.weight` | `[192, 192]` | 36,864 | -0.000153 | 0.020013 | -0.091669 | +0.102101 | 0.0000% |
| `layers.1.attn.k_proj.weight` | `[192, 192]` | 36,864 | -0.000051 | 0.019905 | -0.080969 | +0.075087 | 0.0000% |
| `layers.1.attn.v_proj.weight` | `[192, 192]` | 36,864 | -0.000261 | 0.019888 | -0.083418 | +0.075333 | 0.0000% |
| `layers.1.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000006 | 0.005787 | -0.025150 | +0.023956 | 0.0000% |
| `layers.1.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.1.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | -0.000043 | 0.020026 | -0.095344 | +0.086249 | 0.0000% |
| `layers.1.ffn.up_proj.weight` | `[512, 192]` | 98,304 | +0.000014 | 0.020075 | -0.087648 | +0.084246 | 0.0000% |
| `layers.1.ffn.down_proj.weight` | `[192, 512]` | 98,304 | +0.000004 | 0.005766 | -0.022665 | +0.023703 | 0.0000% |
| `layers.2.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.2.attn.q_proj.weight` | `[192, 192]` | 36,864 | -0.000056 | 0.020033 | -0.091960 | +0.087558 | 0.0000% |
| `layers.2.attn.k_proj.weight` | `[192, 192]` | 36,864 | +0.000026 | 0.019941 | -0.079993 | +0.086330 | 0.0000% |
| `layers.2.attn.v_proj.weight` | `[192, 192]` | 36,864 | -0.000085 | 0.019848 | -0.086697 | +0.083049 | 0.0000% |
| `layers.2.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000015 | 0.005785 | -0.022763 | +0.023820 | 0.0000% |
| `layers.2.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.2.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | +0.000061 | 0.019953 | -0.085773 | +0.083770 | 0.0000% |
| `layers.2.ffn.up_proj.weight` | `[512, 192]` | 98,304 | +0.000079 | 0.020023 | -0.089375 | +0.096196 | 0.0000% |
| `layers.2.ffn.down_proj.weight` | `[192, 512]` | 98,304 | -0.000006 | 0.005771 | -0.024090 | +0.025144 | 0.0000% |
| `layers.3.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.3.attn.q_proj.weight` | `[192, 192]` | 36,864 | +0.000011 | 0.020047 | -0.075175 | +0.081658 | 0.0000% |
| `layers.3.attn.k_proj.weight` | `[192, 192]` | 36,864 | +0.000077 | 0.019989 | -0.074168 | +0.081026 | 0.0000% |
| `layers.3.attn.v_proj.weight` | `[192, 192]` | 36,864 | +0.000147 | 0.019982 | -0.085428 | +0.087464 | 0.0000% |
| `layers.3.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000026 | 0.005764 | -0.026207 | +0.025259 | 0.0000% |
| `layers.3.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.3.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | -0.000054 | 0.019943 | -0.084868 | +0.088153 | 0.0000% |
| `layers.3.ffn.up_proj.weight` | `[512, 192]` | 98,304 | +0.000014 | 0.019942 | -0.087938 | +0.090170 | 0.0000% |
| `layers.3.ffn.down_proj.weight` | `[192, 512]` | 98,304 | +0.000032 | 0.005796 | -0.022011 | +0.024994 | 0.0000% |
| `layers.4.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.4.attn.q_proj.weight` | `[192, 192]` | 36,864 | +0.000117 | 0.020003 | -0.096667 | +0.083438 | 0.0000% |
| `layers.4.attn.k_proj.weight` | `[192, 192]` | 36,864 | +0.000067 | 0.020043 | -0.075847 | +0.081532 | 0.0000% |
| `layers.4.attn.v_proj.weight` | `[192, 192]` | 36,864 | +0.000052 | 0.019922 | -0.082457 | +0.083990 | 0.0000% |
| `layers.4.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000012 | 0.005755 | -0.023020 | +0.022251 | 0.0000% |
| `layers.4.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.4.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | +0.000016 | 0.019995 | -0.083166 | +0.095446 | 0.0000% |
| `layers.4.ffn.up_proj.weight` | `[512, 192]` | 98,304 | +0.000043 | 0.019932 | -0.090262 | +0.090925 | 0.0000% |
| `layers.4.ffn.down_proj.weight` | `[192, 512]` | 98,304 | +0.000025 | 0.005759 | -0.024580 | +0.026031 | 0.0000% |
| `layers.5.norm_1.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.5.attn.q_proj.weight` | `[192, 192]` | 36,864 | -0.000077 | 0.020158 | -0.087587 | +0.090125 | 0.0000% |
| `layers.5.attn.k_proj.weight` | `[192, 192]` | 36,864 | +0.000089 | 0.019854 | -0.080637 | +0.090752 | 0.0000% |
| `layers.5.attn.v_proj.weight` | `[192, 192]` | 36,864 | -0.000019 | 0.020015 | -0.088351 | +0.086724 | 0.0000% |
| `layers.5.attn.out_proj.weight` | `[192, 192]` | 36,864 | +0.000052 | 0.005762 | -0.023581 | +0.024624 | 0.0000% |
| `layers.5.norm_2.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |
| `layers.5.ffn.gate_proj.weight` | `[512, 192]` | 98,304 | -0.000108 | 0.020031 | -0.084802 | +0.080478 | 0.0000% |
| `layers.5.ffn.up_proj.weight` | `[512, 192]` | 98,304 | +0.000018 | 0.019983 | -0.087037 | +0.085757 | 0.0000% |
| `layers.5.ffn.down_proj.weight` | `[192, 512]` | 98,304 | +0.000011 | 0.005780 | -0.025529 | +0.026090 | 0.0000% |
| `final_norm.weight` | `[192]` | 192 | +1.000000 | 0.000000 | +1.000000 | +1.000000 | 0.0000% |

---

## 3. Health & Safety Verification Verdict

- [x] **No Exploding Weights**: All weight initializations remain bounded within $[-0.15, +0.15]$ (well within stable floating-point dynamics).
- [x] **No Vanishing Weights**: Standard deviations are healthy across all layers.
- [x] **No Zero-Weight Dead Zones**: Zero percentages reflect the normal distribution density around exact 0.0 (near 0.000%).
- [x] **Weight Tying Symmetry**: `lm_head.weight` is mathematically identical to `embedding.weight` in memory storage.

**Initialization Verdict**: **VERIFIED HEALTHY**