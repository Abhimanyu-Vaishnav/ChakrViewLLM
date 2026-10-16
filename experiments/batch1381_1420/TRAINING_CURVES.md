# Training Curves and Overfitting Analysis

## Batches 1381–1420: Optimization Dynamics and Validation Tracking

### 1. Training and Validation Progression

The controlled domain adaptation was executed for 80 steps (16 passes over the 21 training documents) with learning rate warmed up to $5.0 \times 10^{-5}$ and cosine decayed to $5.0 \times 10^{-6}$.

| Step | Learning Rate | Train Loss | Validation Loss | Validation Perplexity | Status |
|---|---|---|---|---|---|
| **0 (Baseline)** | 0.00 | 5.4093 | 5.4093 | 223.48 | Pre-training Baseline |
| **5** | $2.75 \times 10^{-5}$ | 4.8206 | 5.3126 | 202.87 | Improved |
| **10** | $5.00 \times 10^{-5}$ | 4.6474 | 5.1455 | 171.65 | Improved |
| **15** | $4.94 \times 10^{-5}$ | 4.4339 | 5.0047 | 149.12 | Improved |
| **20** | $4.78 \times 10^{-5}$ | 4.4939 | 4.9015 | 134.49 | Improved |
| **25** | $4.51 \times 10^{-5}$ | 4.3354 | 4.8223 | 124.25 | Improved |
| **30** | $4.15 \times 10^{-5}$ | 4.1458 | 4.7662 | 117.47 | Improved |
| **35** | $3.73 \times 10^{-5}$ | 3.8200 | 4.7307 | 113.37 | Improved |
| **40** | $3.25 \times 10^{-5}$ | 3.6483 | 4.7022 | 110.19 | Improved |
| **45** | $2.75 \times 10^{-5}$ | 3.6408 | 4.6798 | 107.74 | Improved |
| **50** | $2.25 \times 10^{-5}$ | 3.4389 | 4.6638 | 106.04 | Improved |
| **55** | $1.77 \times 10^{-5}$ | 3.4735 | 4.6514 | 104.74 | Improved |
| **60** | $1.35 \times 10^{-5}$ | 3.7330 | 4.6429 | 103.84 | Improved |
| **65** | $9.91 \times 10^{-6}$ | 3.5660 | 4.6372 | 103.25 | Improved |
| **70** | $7.23 \times 10^{-6}$ | 3.5615 | 4.6326 | 102.79 | Improved |
| **75** | $5.56 \times 10^{-6}$ | 3.6174 | 4.6293 | 102.44 | Improved |
| **80** | $5.00 \times 10^{-6}$ | 3.4277 | **4.6266** | **102.17** | **Best Checkpoint** |

---

### 2. Overfitting Assessment

1. **Train-to-Validation Generalization Gap:**
   - Training loss decreased from ~5.40 to 3.43 ($\Delta = -1.97$ nats).
   - Validation loss on independent development documents decreased monotonically from 5.4093 to 4.6266 ($\Delta = -0.78$ nats, $PPL$ reduced from 223.48 to 102.17).
   - The validation loss curve showed **no signs of turning upward**, confirming that early stopping was not violated and severe memorization was averted by the combination of weight decay (0.05), cosine decay, and gradient clipping (1.0).

2. **Generalization to Strictly Unseen Test Set (27 Held-Out Documents):**
   - On the 27 held-out test documents (3,456 tokens), cross-entropy dropped from **5.0630 to 4.3371** ($\Delta = -0.7259$ nats).
   - Held-out perplexity dropped by **more than half**: from **158.07 down to 76.49**.
   - Top-1 next token accuracy almost doubled: from **12.85% to 23.52%** ($\Delta = +10.67\%$).
   - Top-10 accuracy reached **49.33%** ($\Delta = +8.94\%$).
