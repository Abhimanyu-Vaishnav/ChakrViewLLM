# CR-EXP-16C Capability Scorecard

## 1. Primary Success Gates Status

| Gate | Target Capability | Threshold Metric | Result (Arm A Seed 42) | Result (Arm A Seed 101) | Gate Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Gate 1** | Direct Retrieval | $\ge 70\%$ Exact Match on Held-Out Pairs | **82.0% (164/200)** | **86.5% (173/200)** | **PASSED** |
| **Gate 2** | Dynamic Reassignment | $\ge 60\%$ Exact Match on Overwritten Keys | **99.0% (198/200)** | **99.5% (199/200)** | **PASSED** |
| **Gate 3** | Distractor Resistance | $\ge 50\%$ Exact Match on Noise Prompts | **100.0% (150/150)** | **99.3% (149/150)** | **PASSED** |
| **Gate 4** | Missing Abstention | $\ge 80\%$ Correct Abstention Recall | 58.0% (58/100) | 48.0% (48/100) | **PARTIALLY VALIDATED** |
| **Gate 5** | Unseen Generalization | $\ge 50\%$ Exact Match on Unseen Combos | 0.0% (0/150) | 0.0% (0/150) | **NOT DEMONSTRATED** |

## 2. Capability Registry Update
- **Short-Range Key-Value Retrieval:** **VALIDATED** (84.3% mean across seeds).
- **Dynamic State Update / Reassignment:** **VALIDATED** (99.3% mean across seeds, <0.2% stale errors).
- **Multi-Fact Memory Extraction ($3\dots 5$ facts):** **VALIDATED** (94.7% mean across seeds).
- **Distractor Filtering:** **VALIDATED** (99.7% mean across seeds, 0 distractor captures).
- **Missing Information Abstention:** **PARTIALLY VALIDATED** (53.0% mean recall).
- **Unseen Prompt Template Transfer:** **PARTIALLY VALIDATED** (15.5% mean transfer).
- **Open-Vocabulary Key-Value Binding:** **NOT DEMONSTRATED** (0.0%).
- **Multi-Task Sequence Copying Retention:** **REGRESSED** (0.0% due to catastrophic forgetting).

