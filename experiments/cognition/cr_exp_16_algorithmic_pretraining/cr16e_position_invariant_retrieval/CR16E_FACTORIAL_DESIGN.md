# CR-16E Factorial Experimental Design & Matrix Formulation

### 1. The Core Scientific Objective
CR-EXP-16E addresses the primary cognitive bottlenecks identified during the forensic audit of CR-16D-R:
1. **Absolute Positional Bias:** Single-fact pretraining caused models to lock onto static token offsets, collapsing to 0/100 on permuted multi-fact pairs.
2. **Multi-Token Subword Completion Drift:** Lack of multi-token value span supervision caused generation to drift to familiar lexical attractors after emitting the correct first subword.
3. **Loss of Calibrated Abstention:** Unconstrained copying gradients eroded the negative-sampling signal, reducing abstention on unbound queries to $<20\%$.

---

### 2. The 2×2 Factorial Experimental Matrix

| Arm Identifier | Position Distribution | Value-Span Curriculum | Copy Replay Samples | Retr Samples | Abstain Samples | Total Budget |
|---|---|---|---|---|---|---|
| **Arm A (Control)** | Fixed (Pos 0) | Standard (Len 1–2) | 1,500 | 2,000 | 500 | 4,000 |
| **Arm B (Position Intervention)** | Shuffled Uniformly (Pos 0..3) | Standard (Len 1–2) | 1,500 | 2,000 | 500 | 4,000 |
| **Arm C (Span Intervention)** | Fixed (Pos 0) | Balanced (Len 1..4) | 1,500 | 2,000 | 500 | 4,000 |
| **Arm D (Full Intervention)** | Shuffled Uniformly (Pos 0..3) | Balanced (Len 1..4) | 1,500 | 2,000 | 500 | 4,000 |

### 3. Experimental Controls Held Strictly Constant Across All 12 Runs
- **Common Initialization:** Verified CR-16A Arm D parent checkpoint (`9d609a77...`) across all runs.
- **Model Architecture:** Canonical ChakrMicro v0.1 ($d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$).
- **Optimizer:** AdamW ($lr=2\times 10^{-4}$, weight decay 0.01, linear warmup 40 steps, cosine decay to $2\times 10^{-5}$).
- **Batch Size & Updates:** Batch size 16, 400 optimizer updates.
- **Loss Masking:** Target-only token loss masking (prompt tokens set to `-100`).
- **Seeds:** Evaluated across 3 independent random seeds per arm: `2026`, `3030`, `4040`.

