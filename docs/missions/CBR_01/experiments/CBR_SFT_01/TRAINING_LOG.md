# CBR-SFT-01 — PILOT TRAINING LOG

**Experiment ID:** `CBR-SFT-01`  
**Parent Checkpoint:** Canonical Step 10,000 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Parent Model:** `ChakrMicro v0.1` (3,443,136 parameters)  
**Execution Timestamp:** 2026-11-04T12:19:46+05:30  
**Device:** CPU (Native Intel Core i7-13700H)  
**Total Steps:** 100  
**Effective Batch Size:** 4 (Micro-batch 2, Gradient Accumulation 2)  
**Elapsed Wall Time:** 14.75 seconds  
**Peak RAM Utilization:** 312 MB  

---

## 1. Optimizer & Training Dynamics Progression

| Step | Train Loss | Learning Rate | Val Loss | Val Perplexity | Grad Norm | Step Time (s) | Active Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Init)**| — | 5.00e-05 | 4.6659 | 106.26 | — | — | — |
| **1** | 5.0409 | 5.00e-05 | — | — | 3.69 | 0.142 | 350 |
| **10** | 5.2097 | 4.91e-05 | 4.7655 | 117.39 | 4.23 | 0.138 | 328 |
| **20** | 5.2807 | 4.61e-05 | 4.7740 | 118.39 | 3.54 | 0.140 | 362 |
| **25** | 5.0480 | 4.39e-05 | — | — | 3.01 | 0.136 | 310 |
| **30** | 4.5344 | 4.13e-05 | 5.2126 | 183.57 | 4.17 | 0.135 | 344 |
| **40** | 4.5190 | 3.51e-05 | 4.8619 | 129.27 | 3.68 | 0.141 | 358 |
| **50** | 4.1540 | 2.82e-05 | 4.5068 | 90.63 | 4.05 | 0.139 | 374 |
| **60** | 5.3064 | 2.12e-05 | 4.5600 | 95.58 | 4.05 | 0.134 | 330 |
| **70** | 4.2300 | 1.49e-05 | 4.9954 | 147.73 | 3.85 | 0.138 | 348 |
| **75** | 5.1032 | 1.21e-05 | — | — | 4.40 | 0.137 | 362 |
| **80** | 4.4301 | 9.72e-06 | 4.2448 | 69.74 | 3.61 | 0.139 | 340 |
| **90** | 3.9744 | 6.33e-06 | 4.4771 | 87.98 | 3.85 | 0.142 | 356 |
| **100** | 4.5175 | 5.01e-06 | 4.7554 | 116.21 | 3.11 | 0.140 | 322 |

---

## 2. Key Observations
1. **Stability:** All 100 steps completed with finite loss and bounded gradient norms ($\sim 3.0 - 4.4$), with zero exploding or vanishing gradients.
2. **Validation Fluctuation:** Validation loss fluctuated between $4.24$ and $5.21$, reaching its minimum at step 80 ($4.2448$).
3. **Execution Efficiency:** Highly optimized CPU throughput at $\sim 6.8\text{ steps/sec}$ (~13.6 sequences/sec).
