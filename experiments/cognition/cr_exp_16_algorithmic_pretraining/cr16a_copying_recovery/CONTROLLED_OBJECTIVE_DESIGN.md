# Controlled Objective Design

## 1. Objective Design Matrix
CR-EXP-16A designed and tested four controlled experimental arms to isolate each contributing variable:

| Arm ID | Objective Formulation | Templates | Training Set Size | Optimizer / LR | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm A** | Full-Sequence Loss | Single (`Copy: {s} -> {s}`) | 1,200 | AdamW, $1\times 10^{-4}$ | Exact historical reproduction control |
| **Arm B** | Target-Only Loss (`-100`) | Single (`Copy: {s} -> {s}`) | 1,200 | AdamW, $1\times 10^{-4}$ | Isolate effect of target masking |
| **Arm C** | Target-Only Loss (`-100`) | Multi-Template (6 templates) | 3,000 | AdamW, $3\times 10^{-4}$ | Test template invariance & curriculum scaling |
| **Arm D** | Target-Only Loss (`-100`) | Multi-Template + Diverse single/multi | 5,000 | AdamW, $3\times 10^{-4}$ | Generalization & category prior preservation |

## 2. Gradient Dynamics
Under Target-Only Loss, the parameter update vector is:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \left( -\sum_{\tau \in \text{Target}} \log P(x_\tau \mid x_{<\tau}; \theta) \right)$$
Because no gradients emanate from the prompt positions, the feedforward layers and attention projection matrices ($W_Q, W_K, W_V, W_O$) are updated solely to route the information present in $x_{\le T_{\text{prompt}}}$ to $x_{> T_{\text{prompt}}}$.

