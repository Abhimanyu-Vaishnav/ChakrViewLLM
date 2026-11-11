# CR-16DR Training Comparability & Target-Token Budget

### 1. Training Execution Comparability
All 12 replication runs (Arms A–D $\times$ Seeds 202, 303, 404) were executed under identical system and environmental parameters:
- Device: NVIDIA RTX A1000 Laptop GPU
- Total Steps: 400
- Learning Rate: $2\times 10^{-4}$ with 40-step linear warmup and cosine decay to $2\times 10^{-5}$
- Peak VRAM Usage: $250.1\text{ MB} - 251.3\text{ MB}$ across all runs
- Wall-Clock Duration: $20.4\text{s} - 22.4\text{s}$ per run

---

### 2. Target-Token Exposure by Task

The table below documents the actual count of target tokens supervised (i.e. tokens receiving non-masked loss) during training:

| Arm | Replay Ratio | Supervised Copy Target Tokens | Supervised Retr Target Tokens | Ratio (Copy:Retr) |
|---|---|---|---|---|
| **Arm A (0% Replay)** | 0% | 0 | $30,122 \pm 18$ | 0.00 : 1.00 |
| **Arm B (25% Replay)**| 25% | $13,567 \pm 184$ | $22,668 \pm 121$ | 0.60 : 1.00 |
| **Arm C (50% Replay)**| 50% | $28,039 \pm 218$ | $15,101 \pm 70$ | 1.86 : 1.00 |
| **Arm D (75% Replay)**| 75% | $41,128 \pm 247$ | $7,487 \pm 89$ | 5.49 : 1.00 |

*Notice:* Because sequence copying sequences have longer target token lengths (5 to 12 tokens per target) than key-value retrieval targets (1 to 2 tokens per target), a 50% example replay ratio (Arm C) corresponds to a **1.86:1 target-token ratio** in favor of copying.
This explains why copying retention is so strong even at 25% and 50% example replay.

