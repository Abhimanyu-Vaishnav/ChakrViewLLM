# Stage D — Training Objective & Optimization Audit Report

**Date:** 2026-11-17  
**Mission:** UCIP-09S Controlled Retention & Capability Isolation Study  
**Scripts Audited:**
- `experiments/cognition/ucip_09/train_ucip09.py`
- `experiments/cognition/ucip_09r/train_ucip09r.py`

---

## 1. Executive Summary

This audit compares the training mechanisms, hyperparameters, loss objectives, and optimization schedules between the parent run (**UCIP-09**) and the degraded fine-tuning run (**UCIP-09R**).

The mathematical formulation of the autoregressive causal shift and prompt loss masking in both trainers was verified to be identical and technically sound:
- Causal shift: `shift_logits = logits[:, :-1, :]` vs `shift_targets = tgt[:, 1:]`.
- Target masking: `labels = [-100] * len(prompt_ids) + target_ids`.
- Gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`.

However, severe optimization and protocol defects in the UCIP-09R run led directly to catastrophic forgetting and capability degradation:
1. **Uncalibrated Hyperparameter Aggression on Small Target Corpus:**
   - UCIP-09 trained on 3,240 samples for 125 steps with LR = $2.0 \times 10^{-5}$.
   - UCIP-09R trained on only 885 samples for 160 steps with an *increased* LR = $2.5 \times 10^{-5}$.
   - With batch size 4, 160 steps equals 640 sample exposures over an 885 sample corpus (72.3% epoch pass) at a higher learning rate with 88.5% arithmetic data.
2. **Reduced Regularization / Label Smoothing:**
   - UCIP-09 utilized label smoothing $\epsilon = 0.05$ which softened logits and prevented sharp probability spikes on frequent tokens.
   - UCIP-09R lowered label smoothing to $\epsilon = 0.03$, encouraging the model to become overconfident on repetitive target tokens (`21`, `22`, `cards in total`).
3. **Absence of Validation Anchor Probes:**
   - The validation set in UCIP-09R (`ucip09r_val.jsonl`) suffered from the same severe category skew as `ucip09r_train.jsonl`.
   - Consequently, the model checkpoint selection criterion (`best_val_loss`) selected checkpoints that achieved lower loss solely on arithmetic and copying, actively selecting *against* retention of unrepresented conversational knowledge.

---

## 2. Parameter & Objective Comparison

| Parameter / Configuration | UCIP-09 Parent | UCIP-09R Degraded Run | Impact on Retention |
|---|:---:|:---:|:---:|
| **Parent Checkpoint** | UCIP-08 Medium Candidate | UCIP-09 Medium Candidate | Fine-tuning directly on parent |
| **Learning Rate** | $2.0 \times 10^{-5}$ | $2.5 \times 10^{-5}$ (+25% increase) | Accelerated weight drift away from parent representations |
| **Steps** | 125 | 160 (+28% increase) | Over-exposure to skewed dataset |
| **Batch Size** | 4 | 4 | Identical |
| **Effective Tokens Processed** | ~64,000 tokens | ~81,920 tokens | Higher token volume on narrow domain |
| **Label Smoothing** | $\epsilon = 0.05$ | $\epsilon = 0.03$ (-40% smoothing) | Caused extreme probability peaks on tokens `21`, `22` |
| **Weight Decay** | 0.01 | 0.01 | Identical |
| **Optimizer** | AdamW | AdamW | Identical |
| **Loss Masking** | `-100` on prompts | `-100` on prompts | Verified mathematically correct |
| **Causal Shift** | `logits[:, :-1, :]` | `logits[:, :-1, :]` | Verified mathematically correct |
| **Checkpoint Selection Metric** | Min val loss on balanced val set | Min val loss on 88.5% arithmetic val set | **Actively rewarded conversational forgetting** |

---

## 3. Mathematical Mechanism of Catastrophic Forgetting

In standard fine-tuning under AdamW, the weight update $\Delta \theta$ is given by:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \eta \lambda \theta_t$$

When 88.5% of batches consist of arithmetic templates, the gradient $g_t = \nabla_\theta \mathcal{L}_{\text{arithmetic}}$ acts persistently along the subspace of token prediction for digits and equation syntax. Because the conversational tokens (e.g. `assistant`, `computer`, `स्थानीय`, `हूँ`) receive near-zero positive gradient signal for 160 steps, weight decay $-\eta \lambda \theta_t$ combined with unconstrained gradient updates in the shared multi-head attention projections causes:
1. **Representational drift** in the key-query-value projections $W_q, W_k, W_v$, disrupting attention patterns for natural language prompts.
2. **Logit bias drift** in `lm_head`, shifting base token frequencies to over-index on digits (`21`, `22`) and generic boilerplate (`remaining.`, `total.`).

---

## 4. Corrective Strategy for UCIP-09S Controlled Study

To prevent catastrophic forgetting while improving targeted capabilities:
1. **Balanced Replay Formulation:** The training mixture must preserve an anchor replay ratio of **$\ge 65\%$** general conversational, multilingual, and instruction data, with at most **$35\%$** targeted arithmetic and copying pairs.
2. **Conservative Learning Rate:** Reduce learning rate to $\eta = 1.5 \times 10^{-5}$ or $2.0 \times 10^{-5}$ with warm-up or cosine decay.
3. **Restored Label Smoothing:** Maintain $\epsilon = 0.05$ to suppress single-digit token fixation.
4. **Anchor-Preserving Checkpoint Selection:** Model validation must be evaluated on a balanced validation split containing both conversational retention probes and arithmetic/copying tasks.
