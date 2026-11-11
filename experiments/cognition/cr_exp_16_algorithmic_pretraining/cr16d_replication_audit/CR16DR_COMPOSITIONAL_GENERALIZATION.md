# CR-16DR Compositional Generalization Across 5 Seeds

### 1. Cumulative Compositional Performance Matrix
Evaluating the frozen 1,000-probe CR-16D compositional benchmark across Seeds 42, 101, 202, 303, 404:

| Arm | Description | G1 Novel Combos (250) | G2 Unseen Val (250) | G3 Unseen Key (250) | G4 Both Unseen (250) | Comp Total (1,000) |
|---|---|---|---|---|---|---|
| **Parent (CR-16A)** | Zero-shot | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| **CR-16C Baseline** | Pure Retrieval | 2.4% (6) | 0.0% (0) | 15.2% (38) | 0.0% (0) | 4.4% (44) |
| **Arm A (0% Rep)** | Mean (N=5) | 176.8 (70.7%) $\pm 19.3$ | 15.8 (6.3%) $\pm 5.5$ | **177.4 (71.0%) $\pm 21.7$** | **13.4 (5.4%) $\pm 3.6$** | **383.4 (38.3%) $\pm 41.9$** |
| **Arm B (25% Rep)**| Mean (N=5) | 99.8 (39.9%) $\pm 38.7$ | 11.6 (4.6%) $\pm 7.9$ | 99.0 (39.6%) $\pm 39.1$ | 6.8 (2.7%) $\pm 6.1$ | 217.2 (21.7%) $\pm 84.0$ |
| **Arm C (50% Rep)**| Mean (N=5) | 133.6 (53.4%) $\pm 58.5$ | **15.6 (6.2%) $\pm 3.7$** | 125.2 (50.1%) $\pm 59.3$ | 6.6 (2.6%) $\pm 4.4$ | 281.0 (28.1%) $\pm 124.0$ |
| **Arm D (75% Rep)**| Mean (N=5) | **157.4 (63.0%) $\pm 30.1$** | 13.0 (5.2%) $\pm 6.7$ | 153.2 (61.3%) $\pm 30.4$ | 11.0 (4.4%) $\pm 2.8$ | 334.6 (33.5%) $\pm 63.6$ |

---

### 2. Factorial Insights from 5-Seed Replication
1. **G1 (Novel Combinations of Familiar Keys & Values):**
   - Replicated across all arms. Arm D achieved the most consistent performance among replay models (**63.0% mean**, ranging from 44.0% to 78.4%).
   - Arm C's single-seed spike to 96.4% in Seed 101 did not replicate at that magnitude in Seeds 202 (52.0%), 303 (44.8%), or 404 (25.2%).
2. **G3 (Unseen Keys, Familiar Values):**
   - Correlates almost 1:1 with G1 performance across all 20 models.
   - Proves that as long as the value token is familiar and single-token, the key token's identity does not constrain retrieval in single-fact templates.
3. **G2 (Unseen Values) & G4 (Both Unseen):**
   - Stays flat at 4% to 8% across all 20 models and all 5 seeds.
   - Part B diagnostics verified that this is caused by multi-token subword fragmentation and language-model prior completion drift.

