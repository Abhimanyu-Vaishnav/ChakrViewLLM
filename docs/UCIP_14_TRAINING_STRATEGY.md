# UCIP-14: Training Strategy Design
## Fine-Tuning Dynamics, Loss Formulation, Hyperparameters & Retention Architecture

**Document ID:** UCIP-14-TRAIN-01  
**Project:** ChakrView  
**Status:** DESIGN ONLY — TRAINING NOT AUTHORIZED  

---

## 1. Fine-Tuning Optimization Dynamics

To align ChakrMicro without triggering the catastrophic weight degradation observed in historical runs (UCIP-09R/UCIP-11), the optimization regimen must follow conservative parameter boundaries:

| Parameter | Recommended Specification | Justification |
| :--- | :--- | :--- |
| **Base Model** | UCIP-13 Step 800 (`checkpoint_step_800.pt`) | Proven foundational language representations (loss 4.65) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$) | Standard decoupled weight decay |
| **Weight Decay** | $0.01$ | Prevents large weight deviations |
| **Learning Rate Peak** | $2.0 \times 10^{-5}$ to $3.0 \times 10^{-5}$ | $15\times$ to $25\times$ lower than pretraining ($5 \times 10^{-4}$) to prevent representation collapse |
| **Learning Rate Schedule** | Cosine decay with 10% warmup steps, decaying to $10\%$ min ratio | Smooth adaptation |
| **Gradient Clipping** | Max norm $1.0$ | Protects against destabilizing gradient spikes |
| **Microbatch Size** | 16 sequences | Stable CPU processing |
| **Gradient Accumulation** | 2 steps (Effective Batch Size = 32 sequences) | Increases gradient signal fidelity without RAM overhead |
| **Sequence Horizon** | 512 tokens | Strict context ceiling |
| **Loss Function** | `CrossEntropyLoss(ignore_index=-100)` | **Assistant-only masked loss** |

---

## 2. Loss Formulation: Assistant-Only Supervised Loss

In pretraining (UCIP-13), cross-entropy was calculated across every token in the sequence. In instruction tuning, optimizing the model to predict the *user prompt* forces the model to memorize prompts rather than learn conditional generation.

### Mathematical Objective:
Given an input sequence $X = (x_0, x_1, \dots, x_N)$ where tokens $x_0, \dots, x_P$ comprise the prompt and tokens $x_{P+1}, \dots, x_N$ comprise the assistant response (including the terminal `<EOS>` token):

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{N - P} \sum_{t=P+1}^{N} \log P_\theta(x_t \mid x_{<t})$$

All prompt token positions $t \le P$ receive target label $-100$, completely zeroing out gradients from user tokens:

```python
# Causal Target Masking Implementation (SFTDataset)
input_seq = [BOS] + prompt_tokens + response_tokens
target_seq = ([-100] * len(prompt_tokens)) + response_tokens + [EOS]
```

---

## 3. Foundational Retention & Anti-Catastrophic Forgetting

Fine-tuning exclusively on short question-and-answer pairs causes **catastrophic forgetting**: the model loses general language modeling capabilities, forgets complex syntax, and experiences sharp perplexity explosions on standard text.

### Mitigation Architecture: Controlled Foundational Replay (Mixture Tuning)
To anchor base language representations, the SFT training mixture incorporates a **15% to 20% replay stream** of clean Stage C pretraining text:

```
Batch Composition (Effective Batch = 32):
- 26 sequences: Instruction-Response Pairs (Assistant-only loss)
- 6 sequences: Stage C Pretraining Chunks (Full causal next-token loss)
```

### Dual Early-Stopping Monitors:
Training is monitored across two simultaneous validation metrics:
1. **$L_{\text{SFT, val}}$:** Validation loss on held-out instruction pairs (must decrease).
2. **$L_{\text{Foundational, val}}$:** Validation loss on the official Stage C validation partition (must not drift higher than $+0.25$ over the 4.6859 baseline).

If foundational validation loss exceeds **4.93**, the run is flagged for early termination due to excessive forgetting.
