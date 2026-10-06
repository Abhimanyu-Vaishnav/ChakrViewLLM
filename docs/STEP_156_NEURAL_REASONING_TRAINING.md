# Step 156 — Neural Reasoning Training Experiment

## 1. Experimental Methodology

- **Model:** Isolated ChakrMicro candidate ($3,443,136$ parameters).
- **Execution:** CPU-first PyTorch optimization using standard AdamW and gradient norm clipping ($1.0$).
- **Isolation:** Baseline remains bit-exact and immutable ($\Delta W_{\text{baseline}} = 0$).

## 2. Quantitative Results

| Configuration | Initial Loss | Final Loss | Parameter $\Delta$ Norm | Train Reasoning Acc | Held-Out Reasoning Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU AdamW (lr=1e-3, 35 steps)** | **8.3981** | **1.8913** | **23.6064** | **0.0000** | **0.0000** |

## 3. Analysis

Cross-entropy loss dropped substantially from $8.3981$ to $1.8913$, demonstrating effective parameter updates and gradient backpropagation. However, because the test set evaluates disjoint entities ($X, Y, Z$), zero-shot generalization remains $0.0000$. The model has not yet formed generalizing induction circuits capable of abstract copying without larger pretraining scale or prolonged training.
