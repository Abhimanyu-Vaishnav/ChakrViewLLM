# CR-EXP-14 Phase 5: Candidate Training Results

## 1. Experimental Training Setup
All three candidate models were trained in strict isolation starting from the identical immutable parent:
- **Parent Checkpoint:** CR-12 Candidate B (`cr_exp_12_candidate_b_best_val_copy.pt`, SHA-256: `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e`)
- **Optimization Budget:** 250 optimizer steps per candidate.
- **Hardware & Precision:** Intel CPU, PyTorch float32, single-threaded reproducibility.
- **Batching:** Mini-batch size 8, gradient accumulation 2 (effective batch size 16).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate Schedule:** Linear warmup (20 steps) to $2.5 \times 10^{-5}$, followed by cosine decay to $2.5 \times 10^{-6}$.
- **Gradient Clipping:** Max norm 0.5.

---

## 2. Training Loss & Validation Trajectory

| Step | Candidate A (ESVC) Train Loss | Candidate A Val Loss | Candidate B (ISUC) Train Loss | Candidate B Val Loss | Candidate C (SED) Train Loss | Candidate C Val Loss | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | — | 5.4411 | — | 5.4411 | — | 5.4411 | $2.50 \times 10^{-5}$ |
| **25** | 4.6122 | 3.5643 | 4.6890 | 3.5799 | 5.3428 | 4.8169 | $2.50 \times 10^{-5}$ |
| **50** | 2.7442 | 2.4338 | 2.8274 | 2.4917 | 2.9379 | 4.1991 | $2.41 \times 10^{-5}$ |
| **75** | 1.8791 | 2.0775 | 2.0086 | 2.1226 | 1.6832 | 3.8793 | $2.20 \times 10^{-5}$ |
| **100** | 1.4754 | 1.9048 | 1.5958 | 1.9601 | 1.1219 | **3.8244** | $1.89 \times 10^{-5}$ |
| **125** | 1.2363 | 1.8114 | 1.3295 | 1.8743 | 0.8722 | 3.8263 | $1.53 \times 10^{-5}$ |
| **150** | 1.1030 | 1.7640 | 1.1678 | 1.8268 | 0.7278 | 3.8488 | $1.15 \times 10^{-5}$ |
| **175** | 1.0044 | 1.7362 | 1.0671 | 1.8006 | 0.6484 | 3.8793 | $7.90 \times 10^{-6}$ |
| **200** | 0.9608 | 1.7177 | 1.0278 | 1.7874 | 0.5948 | 3.8988 | $5.02 \times 10^{-6}$ |
| **225** | 0.9146 | 1.7117 | 0.9754 | 1.7793 | 0.5855 | 3.9028 | $3.15 \times 10^{-6}$ |
| **250** | 0.9039 | **1.7072** | 0.9529 | **1.7758** | 0.5707 | 3.9105 | $2.50 \times 10^{-6}$ |

---

## 3. Checkpoint Artifact Registry

| Candidate | Best Val Step | Best Val Loss | Best Val Checkpoint Hash (SHA-256) | Final Checkpoint Hash (SHA-256) |
| :--- | :---: | :---: | :--- | :--- |
| **Candidate A (ESVC)** | 250 | **1.7072** | `58f51bc249aec77235bbdd07179dc130f72e24947500b05fa092f10752a8f033` | `8d5b545591b3e7f9a2cb5dbb817f0d5e13094a9b94b964d797c170be26e0e329` |
| **Candidate B (ISUC)** | 250 | **1.7758** | `e838eb33e6e49cef68e6afb12e7f93bd788f1259940647dce7ef70717f512cf2` | `fb75d743f42a74c890628bbbd9fb6c325ee66bd56c2c3130db6adc03da41eb53` |
| **Candidate C (SED)** | 100 | **3.8244** | `32f1066f4a6b0c652d42991bcc8c1427b63854a55483ffc8a32b726512600d12` | `322afb50369d2e80c982bad4f3de824c6f60631217f353264eaf18d7475551ee` |

### Key Observations
1. **Candidate A and B** exhibited smooth, monotonic convergence on validation loss, dropping from $5.4411$ down to $1.7072$ and $1.7758$ respectively, with zero signs of gradient instability.
2. **Candidate C** achieved very low training loss ($0.5707$) due to structured delimiter regularity, but reached its validation plateau earlier (step 100, val loss $3.8244$) because the validation set did not use delimiters, testing natural language generalization.
