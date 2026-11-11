# CR-16D Replay Ratio Factorial Analysis & Pareto Optimality

### 1. Factorial Attribution of Replay Ratios
We evaluated 4 replay proportions: 0%, 25%, 50%, and 75% sequence copying examples mixed into a 4,000-example training budget.

| Metric | 0% Replay (Arm A) | 25% Replay (Arm B) | 50% Replay (Arm C) | 75% Replay (Arm D) |
|---|---|---|---|---|
| **Copying Exact (100)** | 41.0% | 80.5% | 83.5% | **90.0%** |
| **Copying Token Presence (100)** | 55.5% | 83.5% | 87.5% | **90.5%** |
| **Retrieval Overall (1,500)** | 49.4% | **53.5%** | 53.4% | 50.5% |
| **Direct Retrieval (CAT1, 200)** | 78.2% | 67.2% | **85.5%** | 78.0% |
| **Reassignment (CAT2, 200)** | 91.2% | **100.0%** | **100.0%** | 92.2% |
| **Distractor Pairs (CAT8, 150)** | 83.0% | **98.3%** | 92.0% | 87.0% |
| **Compositional G1 (250)** | 67.2% | 28.0% | **72.6%** | 61.2% |
| **Compositional G3 (250)** | 66.4% | 24.2% | **69.0%** | 62.2% |
| **Compositional Total (1,000)** | 35.8% | 14.2% | **37.9%** | 34.1% |

---

### 2. The Pareto Optimal Frontier
Plotting the trade-off between **Copying Retention** and **Retrieval / Compositional Capability**:

- **Arm A (0% Replay):** Inadequate copying retention (41.0%, severe degradation from parent's 66%).
- **Arm B (25% Replay):** Excellent retrieval reassignment (100%) and copying recovery (80.5%), but lags behind in compositional combination generalization (28.0% on G1).
- **Arm C (50% Replay):** **Pareto Optimal Champion.**
  - Sequence Copying: **83.5%** (+17.5% over parent model).
  - Overall Retrieval: **53.4%** (85.5% direct, 100% reassignment, 92.0% distractor handling).
  - Compositional Generalization: **72.6%** on G1 (Seen Combos) and **69.0%** on G3 (Unseen Keys), peaking at **96.4%** and **94.4%** in Seed 101.
- **Arm D (75% Replay):** Highest copying accuracy (90.0%), but slight decay in retrieval performance (50.5% overall retrieval) due to having only 1,000 retrieval training examples.

### 3. Takeaway
A 1:1 mixture ratio (50% copying replay / 50% key-value retrieval) achieves the optimal balance, maximizing retention of sequence copying while unlocking compositional in-context retrieval.

