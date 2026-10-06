# STEP 212: ASSOCIATION UPDATE / REBINDING

## Objective
Evaluate whether contextual representations dynamically update an association in context (e.g., $B \to Y$ overridden by a subsequent $B \to X$, query $B \to X$).

## Rebinding Conditions Evaluated
- **Case A**: Immediate rebinding ($B \to 2$ and $B \to 3$ query $B$)
- **Case B**: Delayed rebinding (intervening distractors)
- **Case C**: Distractor pairs surrounding updated pair
- **Case D**: Repeated rebinding ($B \to 1 \dots B \to 2 \dots B \to 3$)
- **Case E**: Reverse rebinding ($B \to 1 \dots B \to 2 \dots B \to 1$)

## Empirical Results (Seed 42)
| Rebinding Case | Updated Target Prob | Obsolete Target Prob | Overwrite Ratio | Updated Value Preferred |
| :--- | :---: | :---: | :---: | :---: |
| **Case A (Immediate)** | $2.07 \times 10^{-4}$ | $1.64 \times 10^{-4}$ | **$0.5574$** | **Yes** |
| **Case B (Delayed)** | $2.08 \times 10^{-4}$ | $1.65 \times 10^{-4}$ | **$0.5575$** | **Yes** |
| **Case C (Distractors)** | $2.07 \times 10^{-4}$ | $1.64 \times 10^{-4}$ | **$0.5580$** | **Yes** |
| **Case D (Repeated)** | $2.05 \times 10^{-4}$ | $2.44 \times 10^{-4}$ | **$0.4568$** | No (primacy effect) |
| **Case E (Reverse)** | $2.43 \times 10^{-4}$ | $1.72 \times 10^{-4}$ | **$0.5853$** | **Yes** |

## Summary
- Mean Overwrite Ratio: **$54.28\%$** ($\ge 50\%$).
- Residual Old-Association Interference: **$45.72\%$**.
- **DIAGNOSTIC EVIDENCE**: Contextual representations favor the most recently bound association over obsolete pairings in 4 out of 5 conditions.
