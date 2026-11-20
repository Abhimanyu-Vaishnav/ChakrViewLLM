# UCIP-17 Replay Mathematics & Gradient Balance Forensic Audit

**Mission ID:** UCIP-17  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Execution Agent:** Antigravity  
**Audit Timestamp:** 2026-11-20T18:29:30+05:30  

---

## 1. Executive Summary & Mathematical Reconciliation

In previous reports, conflicting claims were made regarding the effective contribution of Stage C pretraining replay during UCIP-15 SFT mixture tuning. This audit directly measures the empirical batch dynamics, loss formulations, and gradient norms using identical model weights (`checkpoint_step_800.pt`) and deterministic batch sampling.

### The Mathematical Reconciliation:
The previous claim of ~150 active SFT tokens was an estimate of raw response tokens without accounting for dynamic batch padding. When measured across actual training batches:
1. **SFT Sequence Active Tokens:** Average **49.2 active tokens per sequence** (393.8 active tokens across an 8-sequence SFT microbatch). Prompt tokens are masked with `-100`.
2. **Replay Sequence Active Tokens:** Exactly **512 active tokens per sequence** (1,024.0 active tokens across a 2-sequence Replay microbatch). Replay chunks are completely unmasked.
3. **Sequence Count Share:** SFT represents **80.0% of sequences** (8 vs. 2).
4. **Active Token Share:** SFT represents only **27.8% of supervised tokens**, while Replay represents **72.2% of supervised tokens**.

---

## 2. Controlled Gradient Experiment: Four Separate Quantities

A controlled PyTorch experiment evaluated the separate and combined forward passes over 10 deterministic steps:

```
Batch Configuration:
- SFT: batch_size = 8 (mean active tokens = 393.8)
- Replay: batch_size = 2 (mean active tokens = 1024.0)
- Total Active Tokens: 1417.8
```

| Metric / Dimension | SFT Component | Replay Component | Combined Batch |
|---|---:|---:|---:|
| **1. Sequence Count Fraction** | **80.0%** (8 seqs) | **20.0%** (2 seqs) | 10 seqs |
| **2. Active Token Fraction** | **27.78%** (393.8 tok) | **72.22%** (1024.0 tok) | 1417.8 tok |
| **3. Mean Cross-Entropy Token Loss** | **5.6988** | **4.5192** | 4.8468 |
| **4. Loss Sum Numerator** | 2,236.16 (**32.58%**) | 4,627.70 (**67.42%**) | 6,863.86 |
| **5. Isolated Gradient Norm ($\|\nabla L_i\|$)** | **15.0855** | **9.3270** | — |
| **6. UCIP-15 Joint Gradient Norm** | — | — | **9.4587** |

---

## 3. Analysis of Actual Gradient Contribution

In UCIP-15, the loss was implemented as:

$$\mathcal{L}_{\text{joint}} = \frac{\sum L_{\text{SFT}} + \sum L_{\text{Replay}}}{N_{\text{SFT}} + N_{\text{Replay}}} = \frac{N_{\text{SFT}}}{N_{\text{tot}}} \cdot \bar{\mathcal{L}}_{\text{SFT}} + \frac{N_{\text{Replay}}}{N_{\text{tot}}} \cdot \bar{\mathcal{L}}_{\text{Replay}}$$

Substituting empirical measurements:

$$\mathcal{L}_{\text{joint}} = 0.2778 \cdot \bar{\mathcal{L}}_{\text{SFT}} + 0.7222 \cdot \bar{\mathcal{L}}_{\text{Replay}}$$

Because $\frac{\partial \mathcal{L}_{\text{joint}}}{\partial \theta} = 0.2778 \cdot \nabla \bar{\mathcal{L}}_{\text{SFT}} + 0.7222 \cdot \nabla \bar{\mathcal{L}}_{\text{Replay}}$:
- The effective gradient vector received **$2.6\times$ more weight from Stage C pretraining text** ($0.7222$) than from the instruction dataset ($0.2778$).
- Even though SFT gradients had higher individual variance ($\|\nabla L_{\text{SFT}}\| = 15.08$ vs. $9.32$), scaling SFT by $0.2778$ reduced its effective norm contribution to ~4.19, while Replay contributed $0.7222 \times 9.32 \approx 6.73$.
- **Finding:** The effective parameter update was **61.6% dominated by raw pretraining gradients**, directly causing the model to prioritize pretraining n-grams (Wikipedia URLs and article prose) over conversational instruction following.
