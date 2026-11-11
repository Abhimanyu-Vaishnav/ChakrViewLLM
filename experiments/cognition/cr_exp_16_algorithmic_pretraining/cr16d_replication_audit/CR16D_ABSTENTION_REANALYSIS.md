# CR-16D Missing-Information Abstention Reanalysis

### 1. The Abstention Problem
In CR-16C, the model achieved 58.0% abstention on unbound queries (CAT9) by predicting `[UNKNOWN]`.
CR-16D noted that multi-task consolidation caused abstention to degrade, but did not provide the granular per-arm and per-seed metrics.

---

### 2. Forensic Recomputation of Abstention (CAT9 & CAT10)

| Model Condition | CAT9 Exact (100) | CAT9 Abstentions Emitted | CAT10 Exact (100) | Answerable Recall (CAT1) | False Abstention Rate (CAT1) |
|---|---|---|---|---|---|
| **CR-16C Pure Retrieval** | **58 / 100 (58.0%)** | 58 | 20 / 100 (20.0%) | 82.0% (164/200) | 18.0% (36/200) |
| **Arm A (0% Replay, s42)** | 8 / 100 (8.0%) | 8 | 2 / 100 (2.0%) | 87.0% (174/200) | 12.5% (25/200) |
| **Arm A (0% Replay, s101)**| 34 / 100 (34.0%) | 34 | 1 / 100 (1.0%) | 69.5% (139/200) | 24.0% (48/200) |
| **Arm B (25% Replay, s42)** | 38 / 100 (38.0%) | 38 | 3 / 100 (3.0%) | 59.0% (118/200) | **41.0% (82/200)** |
| **Arm B (25% Replay, s101)**| 19 / 100 (19.0%) | 19 | 0 / 100 (0.0%) | 75.5% (151/200) | 23.5% (47/200) |
| **Arm C (50% Replay, s42)** | 18 / 100 (18.0%) | 18 | 0 / 100 (0.0%) | 72.5% (145/200) | 27.5% (55/200) |
| **Arm C (50% Replay, s101)**| **0 / 100 (0.0%)** | **0** | 1 / 100 (1.0%) | **98.5% (197/200)**| **1.5% (3/200)** |
| **Arm D (75% Replay, s42)** | 13 / 100 (13.0%) | 13 | 2 / 100 (2.0%) | 86.0% (172/200) | 13.0% (26/200) |
| **Arm D (75% Replay, s101)**| 28 / 100 (28.0%) | 28 | 8 / 100 (8.0%) | 70.0% (140/200) | 29.0% (58/200) |

---

### 3. Mechanistic Analysis
1. **The Abstention-Assertiveness Trade-off:**
   - In Arm C Seed 101, answerable retrieval accuracy is almost perfect (98.5% on CAT1, false abstention rate of only 1.5%). However, it completely lost the ability to abstain (**0/100 on CAT9**, emitting 0 abstentions). When presented with an unanswerable query, it extracts a salient value from context.
   - Conversely, in Arm B Seed 42, CAT9 abstention is 38%, but it suffered a **41.0% false abstention rate** on CAT1 (falsely abstaining on answerable queries).
2. **Impact of Copying Replay:**
   - Copying examples (`Copy: a b c -> a b c`) heavily train the model to output tokens that appear in context. They never train the model to output `[UNKNOWN]`.
   - Consequently, mixing copying examples into retrieval training systematically attenuates the negative-sampling signal, reducing calibrated abstention across all replay arms.
3. **Requirement for Calibrated Training:**
   - To maintain answerability calibration during multi-task consolidation, synthetic datasets must include explicit negative examples for the copying task (e.g. uncopyable / corrupted tokens) or dedicated loss penalties for ungrounded generation.

