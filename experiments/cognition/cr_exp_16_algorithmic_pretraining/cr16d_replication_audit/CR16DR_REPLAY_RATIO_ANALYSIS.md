# CR-16DR Replay-Ratio Decision & Pareto Analysis

### 1. Multi-Seed Replay Ratio Assessment (N=5 Seeds per Arm)

| Arm | Copying Replay Ratio | Copying Exact (100) | Retr Overall (1500) | Reassignment (200) | Distractors (150) | G1 Combos (250) | Comp Total (1000) |
|---|---|---|---|---|---|---|---|
| **Arm A** | 0% Replay | 42.6% $\pm 6.0$ | 46.6% $\pm 60.5$ | 84.2% $\pm 14.8$ | 85.2% $\pm 24.1$ | **70.7% $\pm 19.3$** | **38.3% $\pm 41.9$** |
| **Arm B** | 25% Replay | 80.8% $\pm 4.6$ | **53.9% $\pm 9.9$** | **100.0% $\pm 0.0$** | **96.9% $\pm 3.1$** | 39.9% $\pm 38.7$ | 21.7% $\pm 84.0$ |
| **Arm C** | 50% Replay | 83.4% $\pm 3.2$ | 53.2% $\pm 26.6$ | **100.0% $\pm 0.0$** | 91.7% $\pm 6.5$ | 53.4% $\pm 58.5$ | 28.1% $\pm 124.0$ |
| **Arm D** | 75% Replay | **88.2% $\pm 1.6$** | 50.3% $\pm 17.1$ | 94.9% $\pm 4.7$ | 88.5% $\pm 5.1$ | 63.0% $\pm 30.1$ | 33.5% $\pm 63.6$ |

---

### 2. Multi-Objective Trade-Off & Decision
In CR-16D, Arm C was declared the Pareto-optimal champion based largely on the exceptional performance of Seed 101. Across the full 5-seed replication study, a more nuanced picture emerges:

1. **Arm A (0% Replay):** Disqualified. While its retrieval and compositional scores are high, it suffers severe catastrophic forgetting of sequence copying ($42.6\%$, failing Gate 1).
2. **Arm B (25% Replay):** The **Retrieval Champion**.
   - Achieves the highest overall retrieval (53.9%), highest distractor filtering (96.9%), perfect reassignment (100.0%), and solid copying retention (80.8%).
   - However, its compositional generalization lags (39.9% on G1).
3. **Arm C (50% Replay):** The **High-Variance Balanced Model**.
   - Solid copying retention (83.4%), perfect reassignment (100.0%), and 53.4% mean on G1.
   - However, G1 performance has extreme variance across seeds ($\sigma = 58.5$, ranging from 25.2% to 96.4%).
4. **Arm D (75% Replay):** The **Stability & Retention Champion**.
   - Achieves the highest copying retention (88.2% with tiny standard deviation $\pm 1.6\%$).
   - Achieves the most consistent compositional generalization among replay models (63.0% mean on G1 $\pm 30.1\%$).
   - Maintains 94.9% reassignment accuracy and 50.3% overall retrieval.

### 3. Final Replay-Ratio Verdict
- For missions where **in-context variable combination generalization and stability** are paramount: **Arm D (75% Replay)** is the superior, most reproducible choice.
- For missions where **pure retrieval and distractor filtering** are prioritized: **Arm B (25% Replay)** is optimal.
- **Arm C (50% Replay)** remains a viable compromise, but cannot be declared unconditionally superior due to substantial seed-level volatility.

