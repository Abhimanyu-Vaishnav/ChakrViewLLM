# CR-16D Dynamic Value Reassignment Analysis

### 1. Objective
Assess the model's capacity to perform in-context state updates by tracking when an entity's value is modified sequentially across context tokens.

### 2. Evaluated Conditions & Benchmarks
In the CR-16C Benchmark:
- **CAT1 (Direct Static Retrieval, 200 probes):** Single binding (`Store: key = val1. Query: key.`)
- **CAT2 (Dynamic Reassignment, 200 probes):** Sequential override (`Store: key = val1. Update: key = val2. Query: key.`)
- **CAT4 (Query Variation with Reassignment, 150 probes):** Alternative phrasing and syntax templates with reassignment.

---

### 3. Quantitative Reassignment Scores

| Model Arm | CAT1 (Direct) | CAT2 (Reassignment) | CAT4 (Reassign Syntax) | Net Reassignment Delta (CAT2 vs CAT1) |
|---|---|---|---|---|
| **CR-16A Arm D (Parent)** | 0.5% (1/200) | 0.0% (0/200) | 1.3% (2/150) | -0.5% |
| **Pure Retrieval (CR-16C)** | 82.0% (164/200) | 99.0% (198/200) | 97.3% (146/150) | **+17.0%** |
| **Arm A (0% Replay, s42)** | 87.0% (174/200) | 98.5% (197/200) | 90.7% (136/150) | **+11.5%** |
| **Arm A (0% Replay, s101)** | 69.5% (139/200) | 84.0% (168/200) | 58.7% (88/150) | **+14.5%** |
| **Arm B (25% Replay, s42)** | 59.0% (118/200) | **100.0% (200/200)** | 88.0% (132/150) | **+41.0%** |
| **Arm B (25% Replay, s101)** | 75.5% (151/200) | **100.0% (200/200)** | 91.3% (137/150) | **+24.5%** |
| **Arm C (50% Replay, s42)** | 72.5% (145/200) | **100.0% (200/200)** | 76.0% (114/150) | **+27.5%** |
| **Arm C (50% Replay, s101)** | **98.5% (197/200)** | **100.0% (200/200)** | 94.0% (141/150) | **+1.5%** |
| **Arm D (75% Replay, s42)** | 86.0% (172/200) | 91.0% (182/200) | 82.7% (124/150) | **+5.0%** |
| **Arm D (75% Replay, s101)** | 70.0% (140/200) | 93.5% (187/200) | 69.3% (104/150) | **+23.5%** |

---

### 4. Recency Bias vs Genuine Dynamic Routing
1. **Perfect Reassignment Execution:**
   - In both Arm B and Arm C across both seeds, reassignment accuracy reached **100.0% (200/200)**. The model reliably outputs `val2` rather than `val1`.
2. **Superiority Over Static Direct Retrieval:**
   - Across almost every arm, CAT2 performance is equal to or higher than CAT1 performance. This phenomenon occurs because the second assignment (`Update: key = val2`) is closer in position to the final query (`Query: key`), benefiting from both positional recency and the causal self-attention mask.
3. **Absence of Stale-State Leakage:**
   - Inspection of error cases in CAT2 indicates that when failures do occur (e.g. in Arm A s101), the model emits format artifacts or `[UNKNOWN]` rather than hallucinating the stale initial value (`val1`).

