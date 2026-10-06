# STEP 213: ASSOCIATION GENERALIZATION

## Objective
Evaluate generalization across randomized layouts and disjoint entity sets:
- **Training Pool**: Keys $\{A, B, C, D, E\}$, Values $\{1, 2, 3, 4, 5\}$
- **Testing Pool**: Keys $\{P, Q, R, S, T\}$, Values $\{6, 7, 8, 9, 0\}$

## 5-Stage Generalization Diagnostic (Seeds 42, 101, 2026)
| Split Condition | Stage A (Repr Margin) | Stage B (Key Match) | Stage C (Assoc Score) | Stage D (Val Routing) | Stage E (Final Token) | Median Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Known Key + Known Val** | $+0.0711$ | $1.0000$ | $0.0187$ | $0.0000$ | **$0.0000$** | $3,772$ |
| **Known Key + Unseen Val** | $+0.0711$ | $0.2000$ | $0.0170$ | $0.0000$ | **$0.0000$** | $3,652$ |
| **Unseen Key + Known Val** | $+0.0450$ | $0.2000$ | $0.0196$ | $0.0000$ | **$0.0000$** | $3,717$ |
| **Unseen Key + Unseen Val** | **$+0.0450$** | **$0.2000$** | **$0.0267$** | **$0.0000$** | **$0.0000$** | **$1,108$** |

## Localization of the Failure Boundary
- **Stages A & B**: Diagnostic representation separation exists ($+0.0450$ margin).
- **Stage D & E**: Value-position routing and final token readout fail completely ($0.0000$).
- **Status**: Disjoint generalization remains **UNPROVEN**.
