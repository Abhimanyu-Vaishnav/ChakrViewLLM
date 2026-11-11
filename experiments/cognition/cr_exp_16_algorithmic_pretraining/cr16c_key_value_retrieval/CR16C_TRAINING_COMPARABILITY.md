# CR-EXP-16C Training Comparability Report

## 1. Experimental Match Matrix

| Parameter / Dimension | Retrieval Arm A (Target-Masked) | Retrieval Arm B (Full-Loss Control) |
| :--- | :---: | :---: |
| **Parent Checkpoint** | Canonical Stage B Step 10k | Canonical Stage B Step 10k |
| **Model Parameters** | 3,440,064 | 3,440,064 |
| **Dataset Substrate** | `cr16c_retrieval_train.json` (3,000 samples) | `cr16c_retrieval_train.json` (3,000 samples) |
| **Validation Substrate** | `cr16c_retrieval_val.json` (500 samples) | `cr16c_retrieval_val.json` (500 samples) |
| **Training Steps** | 400 | 400 |
| **Batch Size** | 16 | 16 |
| **Total Sequences Processed** | 6,400 | 6,400 |
| **Peak Learning Rate** | $3.0\times 10^{-4}$ | $3.0\times 10^{-4}$ |
| **Weight Decay** | 0.01 | 0.01 |
| **Gradient Clipping** | 1.0 | 1.0 |
| **Seeds Evaluated** | 42, 101 | 42, 101 |

## 2. Convergence Comparison Across Seeds
- **Arm A Seed 42:** Final Val Loss: $0.0434$, Target Acc: $98.96\%$, First Token Acc: $95.40\%$.
- **Arm A Seed 101:** Final Val Loss: $0.0425$, Target Acc: $98.75\%$, First Token Acc: $95.00\%$.
- **Arm B Seed 42:** Final Val Loss: $0.3054$, Target Acc: $88.09\%$, First Token Acc: $90.40\%$.
- **Arm B Seed 101:** Final Val Loss: $0.2995$, Target Acc: $87.70\%$, First Token Acc: $90.20\%$.

Both arms converged smoothly. Arm A achieved $\approx 7\times$ lower validation loss ($0.043$ vs $0.302$) and higher target token accuracy, showing that masking prompt syntax accelerates retrieval circuit formation.

