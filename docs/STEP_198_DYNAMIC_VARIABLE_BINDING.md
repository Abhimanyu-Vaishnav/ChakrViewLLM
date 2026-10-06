# STEP 198: DYNAMIC VARIABLE BINDING

## Objective
Re-test dynamic variable binding under explicit role permutation (`query first` vs `query second`) and entity randomization across familiar vs disjoint entities.

## Tasks
- `order A > B query first -> A`
- `order A > B query second -> B`
- `order P > Q query first -> P`
- `order P > Q query second -> Q`
- Random permutations with distractors: `order X > M query second -> M`.

## Empirical Results (Seed 42)
| Entity Set | First Query Acc | Second Query Acc | Overall Role Acc | Median Target Rank |
| :--- | :---: | :---: | :---: | :---: |
| **Familiar Entities** | $0.0000$ | $0.0000$ | **$0.0000$** | $2,379$ |
| **Disjoint Entities** | $0.0000$ | $0.0000$ | **$0.0000$** | $2,379$ |

## Comparison with Historical Waves
- Step 170: $0.5000$ (relied on static positional bias)
- Step 181: $0.5000$ (positional heuristic)
- Step 189: $0.5000$ (copy head compressed rank, but maintained positional bias)
- **Step 198 (Strict Role Invariance)**: **$0.0000$**.
- When prompt templates dynamically invert the relational order (`order A > B` vs `order B > A`), the model fails to extract role bindings dynamically.
- Variable binding generalization remains **UNPROVEN**.
