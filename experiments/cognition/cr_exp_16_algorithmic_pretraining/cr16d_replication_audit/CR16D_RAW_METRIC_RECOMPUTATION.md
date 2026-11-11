# CR-16D Raw Metric Recomputation & Forensic Verification

### 1. Objective & Methodology
To guarantee scientific integrity, the predictions from all historical CR-16D evaluations were independently re-evaluated using exact string matching on raw predictions, verifying that all aggregate counts and percentages in the original report match raw numerators and denominators without rounding artifacts.

---

### 2. Independent Recomputation Matrix (CR-16D Historical Models)

#### Copying Retention (100 Frozen Exact Copy Probes)
- **CR-16A Arm D Parent:** Exact = `66 / 100` (66.0%), Presence = `68 / 100` (68.0%), FirstToken = `100 / 100` (100.0%)
- **CR-16C Pure Retrieval:** Exact = `0 / 100` (0.0%), Presence = `0 / 100` (0.0%), FirstToken = `0 / 100` (0.0%)
- **Arm A (0% Replay, s42):** Exact = `39 / 100` (39.0%), Presence = `45 / 100` (45.0%), FirstToken = `100 / 100` (100.0%)
- **Arm A (0% Replay, s101):** Exact = `43 / 100` (43.0%), Presence = `51 / 100` (51.0%), FirstToken = `100 / 100` (100.0%)
- **Arm B (25% Replay, s42):** Exact = `81 / 100` (81.0%), Presence = `84 / 100` (84.0%), FirstToken = `100 / 100` (100.0%)
- **Arm B (25% Replay, s101):** Exact = `80 / 100` (80.0%), Presence = `81 / 100` (81.0%), FirstToken = `100 / 100` (100.0%)
- **Arm C (50% Replay, s42):** Exact = `83 / 100` (83.0%), Presence = `84 / 100` (84.0%), FirstToken = `100 / 100` (100.0%)
- **Arm C (50% Replay, s101):** Exact = `84 / 100` (84.0%), Presence = `84 / 100` (84.0%), FirstToken = `100 / 100` (100.0%)
- **Arm D (75% Replay, s42):** Exact = `91 / 100` (91.0%), Presence = `91 / 100` (91.0%), FirstToken = `100 / 100` (100.0%)
- **Arm D (75% Replay, s101):** Exact = `89 / 100` (89.0%), Presence = `89 / 100` (89.0%), FirstToken = `100 / 100` (100.0%)

#### Key-Value Retrieval (1,500 Frozen Probes)
| Model | CAT1 (200) | CAT2 (200) | CAT3 (150) | CAT4 (150) | CAT5 (150) | CAT6 (150) | CAT7 (150) | CAT8 (150) | CAT9 (100) | CAT10 (100) | Total (1,500) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parent | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 1500 (0.0%) |
| CR-16C | 164 | 198 | 149 | 146 | 0 | 0 | 0 | 150 | 58 | 20 | 885 / 1500 (59.0%) |
| Arm A s42 | 174 | 197 | 149 | 136 | 0 | 7 | 0 | 145 | 8 | 2 | 818 / 1500 (54.5%) |
| Arm A s101 | 139 | 168 | 125 | 88 | 0 | 5 | 0 | 104 | 34 | 1 | 664 / 1500 (44.3%) |
| Arm B s42 | 118 | 200 | 150 | 132 | 0 | 8 | 0 | 148 | 38 | 3 | 797 / 1500 (53.1%) |
| Arm B s101 | 151 | 200 | 150 | 137 | 0 | 3 | 0 | 147 | 19 | 0 | 807 / 1500 (53.8%) |
| Arm C s42 | 145 | 200 | 150 | 114 | 0 | 3 | 0 | 133 | 18 | 0 | 763 / 1500 (50.9%) |
| Arm C s101 | 197 | 200 | 150 | 141 | 0 | 7 | 0 | 143 | 0 | 1 | 839 / 1500 (55.9%) |
| Arm D s42 | 172 | 182 | 144 | 124 | 0 | 4 | 0 | 132 | 13 | 2 | 773 / 1500 (51.5%) |
| Arm D s101 | 140 | 187 | 137 | 104 | 0 | 7 | 0 | 129 | 28 | 8 | 740 / 1500 (49.3%) |

#### Compositional Benchmark (1,000 Frozen Probes)
| Model | G1 (Novel Combos, 250) | G2 (Unseen Val, 250) | G3 (Unseen Key, 250) | G4 (Both Unseen, 250) | Total Exact (1,000) |
|---|---|---|---|---|---|
| Parent | 0 | 0 | 0 | 0 | 0 / 1000 (0.0%) |
| CR-16C | 6 | 0 | 38 | 0 | 44 / 1000 (4.4%) |
| Arm A s42 | 190 | 11 | 188 | 16 | 405 / 1000 (40.5%) |
| Arm A s101 | 146 | 13 | 144 | 7 | 310 / 1000 (31.0%) |
| Arm B s42 | 41 | 2 | 36 | 0 | 79 / 1000 (7.9%) |
| Arm B s101 | 99 | 10 | 85 | 12 | 206 / 1000 (20.6%) |
| Arm C s42 | 122 | 15 | 109 | 3 | 249 / 1000 (24.9%) |
| Arm C s101 | 241 | 20 | 236 | 12 | 509 / 1000 (50.9%) |
| Arm D s42 | 196 | 26 | 194 | 16 | 432 / 1000 (43.2%) |
| Arm D s101 | 110 | 12 | 117 | 12 | 251 / 1000 (25.1%) |

---

### 3. Forensic Conclusion
1. All claimed metrics in CR-16D were verified against raw prediction logs.
2. The implied second-seed scores from CR-16D are confirmed:
   - Arm C s42 G1 = `122 / 250` (48.8%), s101 G1 = `241 / 250` (96.4%) $\implies$ Mean = `181.5 / 250` (72.6%).
   - Arm C s42 G3 = `109 / 250` (43.6%), s101 G3 = `236 / 250` (94.4%) $\implies$ Mean = `172.5 / 250` (69.0%).
3. The existence of high variance in Arm C between Seed 42 and Seed 101 is authentic and confirmed.

