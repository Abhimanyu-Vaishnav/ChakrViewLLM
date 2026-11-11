# CR-16DR Sequence Copying Retention Across 5 Seeds

### 1. Cumulative 5-Seed Copying Retention Matrix
Evaluating the frozen 100-probe CR-16 exact copying benchmark across Seeds 42, 101, 202, 303, 404:

| Arm | Description | Seed 42 | Seed 101 | Seed 202 | Seed 303 | Seed 404 | Mean Exact (%) | Std Dev | Gate Status ($\ge 70\%$) |
|---|---|---|---|---|---|---|---|---|---|
| **Arm A** | 0% Replay | 39% | 43% | 38% | 54% | 39% | **42.6%** | $\pm 5.95$ | **FAIL (Severe Degradation)** |
| **Arm B** | 25% Replay | 81% | 80% | 83% | 73% | 87% | **80.8%** | $\pm 4.58$ | **PASS** |
| **Arm C** | 50% Replay | 83% | 84% | 88% | 78% | 84% | **83.4%** | $\pm 3.20$ | **PASS** |
| **Arm D** | 75% Replay | 91% | 89% | 87% | 87% | 87% | **88.2%** | $\pm 1.60$ | **PASS (Most Stable)** |

- **Parent Checkpoint (CR-16A Arm D):** 66.0% (66 / 100).
- **Pure Retrieval Baseline (CR-16C):** 0.0% (0 / 100).

---

### 2. Scientific Conclusions on Retention
1. **Replay is Strictly Necessary:**
   - Training purely on retrieval without replay (Arm A) drops copying accuracy from 66% to 42.6% across 5 seeds.
2. **Replay Induces Positive Capability Transfer:**
   - Replay ratios $\ge 25\%$ (Arms B, C, D) consistently enhance sequence copying beyond the parent's initial capability:
     - Arm B: $+14.8\%$ net improvement over parent
     - Arm C: $+17.4\%$ net improvement over parent
     - Arm D: $+22.2\%$ net improvement over parent
3. **Reproducibility Across Seeds:**
   - Zero runs in Arms B, C, or D exhibited catastrophic forgetting or severe performance collapse. Replay consolidation is an extraordinarily robust technique for small language models.

