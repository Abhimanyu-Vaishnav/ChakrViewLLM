# CR-EXP-16C Training Design

## 1. Objective Formulation
CR-EXP-16C evaluated two distinct optimization objectives:
1. **Target-Only Masked Loss (Arm A):**
   $$\mathcal{L}_{\text{target}} = -\frac{1}{T_{\text{target}}} \sum_{t=T_{\text{prompt}}+1}^{T_{\text{total}}} \log P(x_t \mid x_{<t})$$
   All prompt tokens (the stored facts, context, query, and delimiters) are assigned label `-100`, forcing 100% of gradient updates into the retrieved value.
2. **Full-Sequence Loss Control (Arm B):**
   $$\mathcal{L}_{\text{full}} = -\frac{1}{T_{\text{total}}} \sum_{t=1}^{T_{\text{total}}} \log P(x_t \mid x_{<t})$$
   Standard causal language modeling over both context and target.

## 2. Training Hyperparameters
- Model: Canonical ChakrMicro ($3.44\text{M}$ parameters)
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01)
- Learning Rate: $3.0\times 10^{-4}$ with linear warmup and cosine decay
- Gradient Clipping: $1.0$
- Batch Size: 16
- Total Steps: 400
- Parent Weights: Canonical Stage B step 10,000 (`96357705a...`)

