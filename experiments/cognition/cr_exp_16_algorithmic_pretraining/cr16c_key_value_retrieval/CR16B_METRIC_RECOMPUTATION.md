# CR-16B Metric Recomputation Report

## 1. Scope of Recomputation
All raw generation dumps in `cr16b_copying_generalization/eval_outputs/` were independently parsed and re-evaluated using an independent script.

Every prediction string was compared against ground truth targets:
- Exact match: `pred.strip() == target.strip()`
- Token presence: `set(target.strip().split()).issubset(set(pred.strip().split()))`
- First token match: `pred_first_token == target_first_token`

## 2. Recomputed Results vs Claimed Results

| Evaluated Candidate | Total Probes | Recomputed Exact | Claimed Exact | Match? | Recomputed Presence | Claimed Presence | Match? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage B Baseline** | 1,500 | 0 | 0 | **YES** | 0 | 0 | **YES** |
| **CR-16A Arm D** | 1,500 | 508 | 508 | **YES** | 628 | 628 | **YES** |
| **Arm A (Full+Single) Seed 42** | 1,500 | 25 | 25 | **YES** | 170 | 170 | **YES** |
| **Arm A (Full+Single) Seed 101** | 1,500 | 45 | 45 | **YES** | 152 | 152 | **YES** |
| **Arm B (Target+Single) Seed 42** | 1,500 | 158 | 158 | **YES** | 237 | 237 | **YES** |
| **Arm B (Target+Single) Seed 101** | 1,500 | 147 | 147 | **YES** | 240 | 240 | **YES** |
| **Arm C (Full+Multi) Seed 42** | 1,500 | 26 | 26 | **YES** | 185 | 185 | **YES** |
| **Arm C (Full+Multi) Seed 101** | 1,500 | 30 | 30 | **YES** | 208 | 208 | **YES** |
| **Arm D (Target+Multi) Seed 42** | 1,500 | 200 | 200 | **YES** | 439 | 439 | **YES** |
| **Arm D (Target+Multi) Seed 101** | 1,500 | 164 | 164 | **YES** | 430 | 430 | **YES** |

All claimed numerators, denominators, and percentages in CR-16B were verified **bit-for-bit** from raw prediction files without discrepancy.

