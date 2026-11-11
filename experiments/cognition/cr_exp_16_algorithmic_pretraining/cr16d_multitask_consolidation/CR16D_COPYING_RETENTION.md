# CR-16D Sequence Copying Retention Analysis

### 1. Objective
Assess the extent to which sequence-copying capabilities learned in CR-16A were retained, degraded, or enhanced across different replay ratios.

### 2. Copying Benchmark Results (100 Frozen Exact Copy Probes)

| Model Condition | Exact Match (100) | Token Presence (100) | First Token Match (100) | Net Change vs Parent |
|---|---|---|---|---|
| **Parent (CR-16A Arm D)** | 66 / 100 (66.0%) | 74 / 100 (74.0%) | 100 / 100 (100.0%) | Baseline (0.0%) |
| **Pure Retrieval (CR-16C)** | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | **-66.0% (Total Collapse)** |
| **Arm A (0% Replay, s42)** | 39 / 100 (39.0%) | 53 / 100 (53.0%) | 100 / 100 (100.0%) | -27.0% |
| **Arm A (0% Replay, s101)** | 43 / 100 (43.0%) | 58 / 100 (58.0%) | 100 / 100 (100.0%) | -23.0% |
| **Arm B (25% Replay, s42)** | 81 / 100 (81.0%) | 86 / 100 (86.0%) | 100 / 100 (100.0%) | **+15.0%** |
| **Arm B (25% Replay, s101)** | 80 / 100 (80.0%) | 81 / 100 (81.0%) | 100 / 100 (100.0%) | **+14.0%** |
| **Arm C (50% Replay, s42)** | 83 / 100 (83.0%) | 91 / 100 (91.0%) | 100 / 100 (100.0%) | **+17.0%** |
| **Arm C (50% Replay, s101)** | 84 / 100 (84.0%) | 84 / 100 (84.0%) | 100 / 100 (100.0%) | **+18.0%** |
| **Arm D (75% Replay, s42)** | 91 / 100 (91.0%) | 92 / 100 (92.0%) | 100 / 100 (100.0%) | **+25.0%** |
| **Arm D (75% Replay, s101)** | 89 / 100 (89.0%) | 89 / 100 (89.0%) | 100 / 100 (100.0%) | **+23.0%** |

---

### 3. Key Findings on Retention
1. **Zero-Replay Forgetting is Immediate:**
   - In Arm A (0% replay), exact copying dropped from 66% to 41.0% in just 400 steps of retrieval fine-tuning.
   - In CR-16C, longer training without replay resulted in complete collapse (0/100).
2. **Replay Causes "Super-Retention" (Positive Transfer):**
   - In Arms B, C, and D, exact copying accuracy did not merely stay at 66% — it rose significantly to **80.5% - 90.0%**.
   - **Mechanism:** Learning key-value retrieval forces the attention heads to refine their in-context binding and token-position routing. When regular sequence-copying examples are present in the gradient updates, this sharpened routing mechanism directly benefits arbitrary sequence copying.
3. **First-Token Discipline:**
   - Across all replay arms (Arms A through D), first-token match remained at **100/100 (100.0%)**, indicating that the prompt-decoding format was completely preserved.

