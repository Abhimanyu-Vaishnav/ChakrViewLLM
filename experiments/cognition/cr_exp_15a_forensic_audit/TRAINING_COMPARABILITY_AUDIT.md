# CR-EXP-15A Training Comparability Audit
## Examination of Confounding Variables, Training Regimes, and Optimization Lineage

**Mission Identifier:** CR-EXP-15A  
**Auditor:** Autonomous Scientific Auditor & Optimization Engineer  
**Date:** November 2026  

---

## 1. Experimental Conditions Comparison

Inspection of `experiments/cognition/cr_exp_15_architecture_binding/scripts/train_cr15_models.py` and training histories:

| Parameter / Variable | Candidate A | Candidate B | Candidate C | Candidate D (Arm A) | Arch Arm B | Arch Arm C |
|---|---|---|---|---|---|---|
| **Parent Checkpoint** | CR-12 Parent Best | CR-12 Parent Best | CR-12 Parent Best | CR-12 Parent Best | **None (Random Init)** | **None (Random Init)** |
| **Model Width ($d_{\text{model}}$)**| 192 | 192 | 192 | 192 | 192 | **256** |
| **Layers ($N_{\text{layers}}$)** | 6 | 6 | 6 | 6 | 6 | 6 |
| **Attention Heads** | 6 | 6 | 6 | 6 | **3** | **4** |
| **Head Dimension ($d_{\text{head}}$)**| 32 | 32 | 32 | 32 | **64** | **64** |
| **Parameter Count** | 3,443,136 | 3,443,136 | 3,443,136 | 3,443,136 | 3,443,136 | **5,767,424** |
| **Curriculum Dataset** | General LM (500) | Copy & Select (500)| Static Binding (500)| Dynamic Updates (500)| Dynamic Updates (500)| Dynamic Updates (500)|
| **Validation Dataset** | General Val (100) | General Val (100) | General Val (100) | State Update Val (100)| State Update Val (100)| State Update Val (100)|
| **Training Steps** | 200 | 200 | 200 | 200 | 200 | 200 |
| **Effective Batch Size** | 16 | 16 | 16 | 16 | 16 | 16 |
| **Peak Learning Rate** | 2.5e-5 | 2.5e-5 | 2.5e-5 | 2.5e-5 | 2.5e-5 | 2.5e-5 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| **Random Seed** | 42 | 42 | 42 | 42 | 42 | 42 |

---

## 2. Confounding Variables Identified

The audit identifies **two major experimental confounders** in the CR-15 study design:

### Confounder 1: Checkpoint Initialization vs Random Initialization
- **Candidate D** was initialized from the pretrained **CR-12 Parent checkpoint** (which already possessed conversational and language representations across tens of thousands of prior training steps).
- **Arch Arm B and Arch Arm C** were initialized **from scratch (random weights)** because their layer dimensions ($d_{\text{head}}=64$, width 256) could not directly load the $192/32$ parent state dict.
- **Scientific Impact:** Comparing Candidate D directly against Arm B and Arm C conflated *architectural effects* with *pretraining state*. Training a transformer from scratch for only 200 steps (3,200 sequences) at learning rate $2.5 \times 10^{-5}$ is vastly insufficient for random weights to converge on basic English grammar, explaining why Arm B collapsed to `Assistant:` and Arm C collapsed to `AAAA`.

### Confounder 2: Incompatible Validation Loss Evaluators
- Candidates A, B, and C were evaluated on `data/val.json` (a mixed conversational validation set), achieving validation losses between $4.96$ and $5.12$.
- Candidate D was evaluated on a dynamic update validation subset formatted identically to its training set, achieving a loss of $1.84$.
- The lower loss of Candidate D was mistakenly interpreted in early stages as superior generalization, whereas it was actually evaluating on a different, templated distribution.

