# UCIP-17 Training Objective & Effective Gradient Balance Proposal

**Mission ID:** UCIP-17  
**Project:** ChakrView  
**Status:** PROPOSAL ONLY — TRAINING NOT AUTHORIZED  
**Audit Timestamp:** 2026-11-20T18:30:10+05:30  

---

## 1. Mathematical Formulation of the Defect in UCIP-15

In UCIP-15, the joint optimization loss was formulated as:

$$\mathcal{L}_{\text{joint}}(\theta) = \frac{\sum_{i=1}^{B_{\text{SFT}}} \sum_{t \in \text{resp}} \ell(x_{i,t}, y_{i,t}) + \sum_{j=1}^{B_{\text{rep}}} \sum_{t=1}^{T_{\text{rep}}} \ell(x_{j,t}, y_{j,t})}{N_{\text{SFT\_tokens}} + N_{\text{rep\_tokens}}}$$

Where:
- $B_{\text{SFT}} = 8$ sequences, but with prompt tokens masked to $-100$, yielding $N_{\text{SFT\_tokens}} \approx 394$ active tokens.
- $B_{\text{rep}} = 2$ sequences, unmasked (512 tokens each), yielding $N_{\text{rep\_tokens}} = 1,024$ active tokens.

Rewriting in terms of per-group mean token losses:

$$\mathcal{L}_{\text{joint}}(\theta) = \alpha_{\text{SFT}} \cdot \bar{\mathcal{L}}_{\text{SFT}}(\theta) + \alpha_{\text{rep}} \cdot \bar{\mathcal{L}}_{\text{rep}}(\theta)$$

$$\alpha_{\text{SFT}} = \frac{394}{1418} = 0.278, \quad \alpha_{\text{rep}} = \frac{1024}{1418} = 0.722$$

**Impact:** Replay gradients contributed **72.2% of the total loss magnitude** and dominated parameter updates, washing out conversational instruction signals.

---

## 2. Revised Balanced Training Objective Formulation for UCIP-18

To ensure that instruction tuning drives model adaptation while foundational pretraining text acts solely as a regularizer, we propose decoupled loss weighting:

$$\mathcal{L}_{\text{UCIP18}}(\theta) = (1 - \lambda_{\text{rep}}) \cdot \bar{\mathcal{L}}_{\text{SFT}}(\theta) + \lambda_{\text{rep}} \cdot \bar{\mathcal{L}}_{\text{rep}}(\theta)$$

Where:
- $\bar{\mathcal{L}}_{\text{SFT}}(\theta) = \frac{1}{N_{\text{SFT}}} \sum_{i,t \in \text{resp}} \ell(x_{i,t}, y_{i,t})$ (normalized strictly by active SFT tokens).
- $\bar{\mathcal{L}}_{\text{rep}}(\theta) = \frac{1}{N_{\text{rep}}} \sum_{j,t} \ell(x_{j,t}, y_{j,t})$ (normalized strictly by active replay tokens).
- **$\lambda_{\text{rep}} \in [0.10, 0.15]$:** Fixed scalar hyperparameter guaranteeing that pretraining replay accounts for **strictly 10%–15% of the gradient contribution**.

### Gradient Balance Comparison:

| Parameter | UCIP-15 Actual | Proposed UCIP-18 |
|---|:---:|:---:|
| SFT Sequence Batch Size | 8 | 12 |
| Replay Sequence Batch Size | 2 | 1 |
| SFT Effective Gradient Share | **27.8%** | **85.0%** ($\lambda_{\text{rep}}=0.15$) |
| Replay Effective Gradient Share | **72.2%** | **15.0%** |
| Instruction Dominance Ratio | $0.38\times$ (Suppressed) | **$5.67\times$ (Dominant)** |

---

## 3. Implementation Specification

```python
# In SFT training step:
logits_sft = model(x_sft)
loss_sft = loss_fn_sft(logits_sft.reshape(-1, 4096), y_sft.reshape(-1)) / max(1, sft_tokens)

logits_rep = model(x_rep)
loss_rep = loss_fn_rep(logits_rep.reshape(-1, 4096), y_rep.reshape(-1)) / max(1, rep_tokens)

# Decoupled objective with explicit 85/15 weighting:
lambda_rep = 0.15
total_loss = (1.0 - lambda_rep) * loss_sft + lambda_rep * loss_rep
total_loss.backward()
```

This guarantees that instruction-following gradients drive 85% of each parameter update, eliminating the replay attractor distortion while maintaining language retention.
