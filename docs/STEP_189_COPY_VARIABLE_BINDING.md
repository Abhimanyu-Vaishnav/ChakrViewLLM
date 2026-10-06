# STEP 189: VARIABLE BINDING WITH COPY MECHANISM

## Objective
Re-evaluate dynamic variable binding (`order A > B query first -> A`, `order A > B query second -> B`) after equipping the network with the neural pointer head.

## Experimental Setup
- **Familiar entities**: Pairs drawn from $\{A, B, C, D, E, F\}$.
- **Disjoint entities**: Pairs drawn from $\{P, Q, R, S\} \times \{X, Y, Z, W\}$.
- **Query roles**: Position-invariant evaluation of `first` vs `second`.

## Comparative Results
| Architecture | Familiar First Acc | Familiar Second Acc | Familiar Overall Acc | Disjoint First Acc | Disjoint Second Acc | Disjoint Overall Acc | Source Pos Acc | Median Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Copy Baseline (Step 181)** | $0.5000$ | $0.5000$ | $0.5000$ | $0.0000$ | $0.0000$ | $0.0000$ | N/A | $3,120$ |
| **Post-Copy Hybrid (Step 189)** | $0.5000$ | $0.5000$ | $0.5000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.1500$ | $1,240$ |

## Mechanistic Analysis
The pointer head improves target rank and assigns attention probability mass to contextual tokens, but the model maintains a $50\%$ positional prior on familiar variables and does not transfer variable binding zero-shot to disjoint entities ($0.0000$). Variable binding generalization beyond in-distribution entities remains **UNPROVEN**.
