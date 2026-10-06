# STEP 199: ITERATIVE RETRIEVAL FOR MULTI-HOP REASONING

## Objective
Evaluate whether the retrieval circuit can execute chained multi-hop reasoning across 1 to 4 relational hops using disjoint symbols without symbolic lookup.

## Chained Relations Specification
- **1-hop**: `map |P| -> |Q| query |P| -> Q`
- **2-hop**: `map |P| -> |Q| and |Q| -> |R| query |P| -> R`
- **3-hop**: `map |P| -> |Q| and |Q| -> |R| and |R| -> |S| query |P| -> S`
- **4-hop**: `map |P| -> |Q| and |Q| -> |R| and |R| -> |S| and |S| -> |T| query |P| -> T`

## Empirical Results (Seed 42)
| Relational Hop | Output Accuracy | Target Probability | Median Rank | Hop 1 Intermediate Accuracy | Error Accumulation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Hop** | **$0.0000$** | $2.02 \times 10^{-4}$ | $2,909$ | $0.3333$ | $0.0000$ |
| **2-Hop** | **$0.0000$** | $1.85 \times 10^{-4}$ | $3,433$ | $0.0000$ | $1.0000$ |
| **3-Hop** | **$0.0000$** | $1.83 \times 10^{-4}$ | $3,373$ | $0.0000$ | $1.0000$ |
| **4-Hop** | **$0.0000$** | $2.48 \times 10^{-4}$ | $1,270$ | $0.0000$ | $1.0000$ |

## Failure Boundary Analysis
- **Failure Boundary**: Strictly at **Hop 2**.
- When evaluating 2-hop chains, the model does not propagate the intermediate bridge token ($Q$) into a secondary query representation. Intermediate retrieval accuracy drops to $0.0000$, resulting in complete compounding error failure.
- Multi-hop reasoning remains **REFUTED** without sequential recurrence or multi-step cognitive deliberation.
