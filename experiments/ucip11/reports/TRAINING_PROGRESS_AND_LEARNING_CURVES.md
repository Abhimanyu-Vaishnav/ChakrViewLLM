# UCIP-11 Phase G: Training Progress & Learning Curves

**Mission ID:** UCIP-11  
**Experiment Runs:** Conditions B, C, D, and E  
**Parent Baseline:** UCIP-09 Medium (`5513fefe...`)  
**Artifact Directory:** `experiments/ucip11/results/training_summary.json`  

---

## 1. Executive Summary

Four controlled training runs were executed deterministically on CPU to compare continuous pretraining on clean natural language (CPT), supervised instruction tuning (SFT), two-stage training (CPT + SFT), and from-scratch language acquisition.

---

## 2. Quantitative Training Metrics

| Condition ID | Description | Starting State | Total Steps | Initial Train Loss | Final Train Loss | Best Val Loss | Elapsed (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| **Condition B** | Clean Stage B CPT | UCIP-09 Medium | 100 | 6.4904 | **4.2590** | 6.8665 | 17.7s | 2,899 |
| **Condition C** | Clean Exp34 v3 SFT | UCIP-09 Medium | 100 | 6.1528 | **5.3849** | **5.2047** | 26.4s | 1,939 |
| **Condition D** | CPT + SFT (2-Stage)| Condition B Best | 60 | 6.0124 | **5.6458** | 5.6049 | 16.2s | 1,896 |
| **Condition E** | Scratch Pretraining| Random Init (Seed 42) | 100 | 7.1910 | 5.7256 | 7.6345 | 19.8s | 2,586 |

---

## 3. Loss Trajectories

### Condition B (Continuous Pretraining on Clean Multi-Domain Text)
- Step 20: Train Loss = 6.4904 | Val Loss = 7.0344
- Step 40: Train Loss = 5.5206 | Val Loss = 6.9591
- Step 60: Train Loss = 4.5919 | Val Loss = 6.9954
- Step 80: Train Loss = 4.7254 | Val Loss = 6.9884
- Step 100: Train Loss = **4.2590** | Val Loss = **6.8665**

### Condition C (Instruction Tuning on Exp34 v3)
- Step 20: Train Loss = 6.1528 | Val Loss = 6.0173
- Step 40: Train Loss = 5.8158 | Val Loss = 5.7165
- Step 60: Train Loss = 5.6300 | Val Loss = 5.5834
- Step 80: Train Loss = 5.5471 | Val Loss = 5.3408
- Step 100: Train Loss = **5.3849** | Val Loss = **5.2047**

### Condition D (Two-Stage CPT followed by SFT)
- Step 20: Train Loss = 6.0124 | Val Loss = 5.9629
- Step 40: Train Loss = 5.7702 | Val Loss = 5.7224
- Step 60: Train Loss = **5.6458** | Val Loss = **5.6049**

### Condition E (From Scratch Random Weights)
- Step 20: Train Loss = 7.1910 | Val Loss = 7.6345
- Step 40: Train Loss = 6.3562 | Val Loss = 7.6674
- Step 60: Train Loss = 5.8982 | Val Loss = 7.8928
- Step 80: Train Loss = 6.1617 | Val Loss = 8.1548
- Step 100: Train Loss = 5.7256 | Val Loss = 8.0013 (Clear Overfitting on Train Slice)

---

## 4. Key Training Observations

1. **Condition C Achieved Strongest Convergence:** Supervised instruction tuning on Exp34 v3 drove validation cross-entropy down to **5.2047**, the lowest across all runs.
2. **CPT Generalization Gap:** In Condition B, train loss decreased steadily from 6.49 to 4.25, while validation loss leveled off around 6.86–6.95.
3. **Condition E Scratch Bottleneck:** 100 steps of random initialization from scratch on CPU was insufficient to acquire grammatical subword syntax, resulting in un-converged token distributions (`___`).
