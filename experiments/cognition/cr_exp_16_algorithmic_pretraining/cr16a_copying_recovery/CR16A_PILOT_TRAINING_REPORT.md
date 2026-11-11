# CR-EXP-16A Pilot Training Report

## 1. Training Setup & Hyperparameters
- Base Architecture: ChakrMicro ($d_{\text{model}}=192, 6\text{ layers}, 6\text{ heads}, d_{\text{head}}=32, 3.44\text{M params}$)
- Initial Checkpoint: Canonical Stage B step 10,000 (`96357705a...`)
- Precision: PyTorch Float32
- Hardware: CUDA GPU (NVIDIA RTX 3050 Laptop GPU)
- Batch Size: 16
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999, \text{weight\_decay}=0.01$)
- Learning Rate: $3.0\times 10^{-4}$
- Gradient Clipping: $1.0$

## 2. Training Run Summaries

### Arm C (Multi-Template Curriculum, 400 steps)
- Step 25: Train Loss $2.5660$
- Step 50: Train Loss $1.6753$ | Val Loss $1.7328$ | Target Acc $59.59\%$ | First Token Acc $36.67\%$
- Step 100: Train Loss $1.2028$ | Val Loss $1.1540$ | Target Acc $70.99\%$ | First Token Acc $74.33\%$
- Step 200: Train Loss $0.4846$ | Val Loss $0.5966$ | Target Acc $83.16\%$ | First Token Acc $97.00\%$
- Step 300: Train Loss $0.4076$ | Val Loss $0.3672$ | Target Acc $90.17\%$ | First Token Acc $98.33\%$
- Step 400: Train Loss $0.1087$ | Val Loss $0.2119$ | Target Acc $94.88\%$ | First Token Acc $97.67\%$

### Arm D (Generalization Pretraining, 500 steps)
- Step 50: Train Loss $2.2014$ | Val Loss $2.1428$ | Target Acc $53.52\%$ | First Token Acc $29.67\%$
- Step 100: Train Loss $1.4088$ | Val Loss $1.5740$ | Target Acc $62.33\%$ | First Token Acc $64.33\%$
- Step 200: Train Loss $0.7153$ | Val Loss $0.8359$ | Target Acc $78.67\%$ | First Token Acc $95.67\%$
- Step 300: Train Loss $0.4238$ | Val Loss $0.4786$ | Target Acc $88.06\%$ | First Token Acc $96.33\%$
- Step 400: Train Loss $0.1892$ | Val Loss $0.2534$ | Target Acc $93.40\%$ | First Token Acc $99.00\%$
- Step 500: Train Loss $0.2102$ | Val Loss $0.2091$ | Target Acc $95.06\%$ | First Token Acc $99.67\%$

Both runs converged smoothly without gradient explosions or numerical instability.

