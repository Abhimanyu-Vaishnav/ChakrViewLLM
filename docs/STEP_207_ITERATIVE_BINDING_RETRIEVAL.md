# STEP 207: BINDING → ITERATIVE RETRIEVAL

## Objective
Evaluate whether contextual neural binding supports iterative multi-hop reasoning across 1 to 4 hops using disjoint symbols without symbolic lookup.

## Gate Status
- **Precondition Gate (Step 205)**: Disjoint associative binding did not meet the $\ge 0.50$ promotion threshold (observed $0.0000$).
- **Gated Protocol Executed**: Diagnostic evaluation completed; no claim of emergent multi-hop reasoning.

## Diagnostic Metrics Across Hops (Seed 42)
| Relational Hop | Value Retrieval Acc | Target Probability | Median Target Rank | Query Matching Conf |
| :---: | :---: | :---: | :---: | :---: |
| **Hop 1** | $0.0000$ | $2.02 \times 10^{-4}$ | $2,909$ | $0.3333$ |
| **Hop 2** | **$0.0000$** | $1.85 \times 10^{-4}$ | $3,433$ | $0.0000$ |
| **Hop 3** | **$0.0000$** | $1.83 \times 10^{-4}$ | $3,373$ | $0.0000$ |
| **Hop 4** | **$0.0000$** | $2.48 \times 10^{-4}$ | $1,270$ | $0.0000$ |

## Failure Boundary Diagnosis
- **Failure Boundary**: Strictly at **Hop 1 / Hop 2**.
- Because the network does not transfer intermediate bridge representations into active query states, intermediate retrieval remains $0.0000$, resulting in total compounding error accumulation ($1.0000$).
- Multi-hop reasoning is **REFUTED** under pure single-pass neural execution.
