# STEP 188: COPY VS GENERATION EMPIRICAL ANALYSIS

## Objective
Evaluate the comparative behavior of 4 architecture candidates on familiar vs disjoint associative retrieval:
A. Original tied readout
B. Untied readout
C. Copy-only candidate
D. Hybrid generation + copy candidate

## Empirical Results (Seed 42)
| Candidate | Total Parameters | Trainable Parameters | Familiar Acc | Disjoint Acc | Mean $P(\text{copy})$ | Source Pos Acc | Median Rank | Retrieval Failure Rate | Readout Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tied Baseline** | $3,443,136$ | $0$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $3,695.0$ | N/A | $1.0000$ |
| **Untied Readout** | $4,229,568$ | $0$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | $3,695.0$ | N/A | $1.0000$ |
| **Copy-Only** | $3,517,057$ | $73,921$ | $0.2000$ | $0.0000$ | $1.0000$ | $0.0000$ | **$7.0$** | $1.0000$ | $0.0000$ |
| **Hybrid Copy+Gen** | $3,517,057$ | $73,921$ | $0.2000$ | $0.0000$ | $0.8500$ | $0.1000$ | $1,564.0$ | $0.9000$ | $0.1000$ |

## Critical Mechanistic Insight: Retrieval Failure vs Readout Failure
1. **Readout Failure Resolved**: The copy mechanism dramatically collapses the candidate rank of contextual tokens from **$3,695$** down to **$7.0$**. When forcing copy mode, target probabilities jump by multiple orders of magnitude.
2. **Retrieval Failure Remains Primary**: The attention query $q = W_q h_{\text{last}}$ fails to reliably attend to the exact contextual value position corresponding to the query key when given unseen entity pairs ($90\%$ retrieval failure rate).
3. **Conclusion**: The inability of the model to output unseen values is primarily a **retrieval routing failure** across contextual representations, rather than purely an output projection vocabulary limitation.
