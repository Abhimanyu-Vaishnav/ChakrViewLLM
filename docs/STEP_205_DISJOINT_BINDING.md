# STEP 205: DISJOINT ASSOCIATIVE BINDING (PRIMARY I3 GATE)

## Objective
Evaluate whether associative binding generalizes when keys and values are completely disjoint:
- **Training Identities**: Keys $\{A, B, C, D, E\}$, Values $\{1, 2, 3, 4, 5\}$
- **Testing Identities**: Keys $\{P, Q, R, S, T\}$, Values $\{6, 7, 8, 9, 0\}$

## 4-Way Condition Results (Seeds 42, 101, 2026)
| Split Condition | Stage A (Key Matching) | Stage B (Assoc Score) | Stage C (Value Retrieval) | Stage D (Final Token) | Target Probability | Median Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Known Key + Known Value** | $1.0000$ | $0.2014$ | $0.0000$ | $0.0000$ | $2.01 \times 10^{-4}$ | $2,840$ |
| **Known Key + Unseen Value** | $0.0000$ | $0.1852$ | $0.0000$ | $0.0000$ | $1.85 \times 10^{-4}$ | $3,120$ |
| **Unseen Key + Known Value** | $0.0000$ | $0.1940$ | $0.0000$ | $0.0000$ | $1.94 \times 10^{-4}$ | $2,950$ |
| **Unseen Key + Unseen Value** | **$0.0000$** | **$0.1820$** | **$0.0000$** | **$0.0000$** | **$1.82 \times 10^{-4}$** | **$3,250$** |

## I3 Gate Evaluation
- Stage D accuracy on `unseen_unseen`: **$0.0000$** (Threshold required: $\ge 0.5000$).
- Promotion to Level I3 is **REFUSED**.
- **I3 Status**: **UNPROVEN**.
