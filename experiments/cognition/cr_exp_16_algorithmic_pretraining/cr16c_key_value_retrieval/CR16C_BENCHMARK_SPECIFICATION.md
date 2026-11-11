# CR-EXP-16C Benchmark Specification

## 1. Overview of the Benchmark
The CR-EXP-16C Benchmark (`cr16c_sealed_benchmark_1500.json`, SHA-256: `033f807ba518c168...`) is an independent, frozen evaluation suite of exactly **1,500 probes** designed to evaluate contextual key-value retrieval, dynamic reassignment, distractor resistance, and calibrated abstention.

## 2. Category Distribution & Design Specifications

| Category Code | Category Name | Probe Count | Context Structure | Challenge Evaluated |
| :--- | :--- | :---: | :--- | :--- |
| **CAT1** | Direct Retrieval (Familiar) | 200 | 1–2 stored facts (`Store: city = Delhi...`) | Baseline key-value association |
| **CAT2** | Reassigned Values | 200 | Stored key overwritten (`Update: k = v2`) | Dynamic state update / recency |
| **CAT3** | Counterfactual Swapped Updates | 150 | Swapped values across updates | Rejection of stale values |
| **CAT4** | Multiple Stored Facts | 150 | 3–5 interleaved stored facts | Scaled multi-pair memory lookup |
| **CAT5** | Unseen Key-Value Combos | 150 | Familiar keys paired with unseen values | Compositional generalization |
| **CAT6** | Unseen Keys, Familiar Values | 150 | Held-out keys (`fabric`, `weapon`, `season`) | Open-vocabulary key binding |
| **CAT7** | Familiar Keys, Unseen Values | 150 | Greek / novel values paired with keys | Out-of-distribution value retrieval |
| **CAT8** | Distractor-Heavy Contexts | 150 | Noise notices (`Notice: noise is Alpha`) | Selectivity against unrelated facts |
| **CAT9** | Missing-Key Abstention | 100 | Query key NOT in stored facts | Calibration (Target: `[UNKNOWN]`) |
| **CAT10** | Unseen Serialization Templates | 100 | Unseen format (`Database entry: [k := v]`) | Invariance to prompt syntax |

Total Probes: **1,500**.
Sealed Hash: `033f807ba518c168e095c1bebe68b1e2bf7c743ba0cc8af2320a1ffc334dceae`.

