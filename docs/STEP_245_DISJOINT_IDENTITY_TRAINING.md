# STEP 245: Disjoint Identity Training & I3 Evaluation

## Mission
Execute the primary capability-growth and I3 gate evaluation step across 4 orthogonal identity splits and multiple deterministic seeds.

## Evaluation Protocol
Implemented in [`chakrview/cognition/generalized_disjoint_training.py`](file:///d:/Project/ChakrView/chakrview/cognition/generalized_disjoint_training.py):
- **Orthogonal Splits**:
  1. `known_known`: In-distribution keys and values
  2. `known_unseen`: Familiar keys mapped to completely unseen values
  3. `unseen_known`: Unseen keys mapped to familiar values
  4. `unseen_unseen`: Completely unseen keys and completely unseen values
- **Deterministic Seeds Tested**: `42`, `101`, `2026`.

## Multi-Seed Results
| Split | Seed 42 Acc | Seed 101 Acc | Seed 2026 Acc | Mean Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **known / known** | 10.0% | 20.0% | 20.0% | 16.7% |
| **known / unseen** | 0.0% | 0.0% | 0.0% | 0.0% |
| **unseen / known** | 10.0% | 10.0% | 10.0% | 10.0% |
| **unseen / unseen** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

## I3 Promotion Gate Determination
- **Criterion**: Requires unseen/unseen retrieval accuracy $\ge 0.50$ across seeds for `I3_CANDIDATE_ACHIEVED`.
- **Observed Result**: Unseen/unseen accuracy is $0.0000$ across all seeds.
- **Official Designation**: `I3_NOT_ACHIEVED`.
- **Baseline Integrity**: Canonical baseline remains bit-exact (`c5571c...a282da`).
