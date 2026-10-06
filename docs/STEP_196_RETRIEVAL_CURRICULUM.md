# STEP 196: RETRIEVAL CIRCUIT CURRICULUM

## Objective
Evaluate the learnability of associative retrieval across a 9-level staged curriculum from trivial single-pair recall to complex randomized disjoint retrieval.

## Curriculum Levels & Performance Summary
| Level | Description | Key Matching Acc | Value Position Acc | Promotion Threshold ($\ge 0.50$) | Level Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **R0** | Single pair | $1.0000$ | $1.0000$ | Met | **PASSED** |
| **R1** | Two pairs | $0.4000$ | $0.2000$ | Unmet | FAILED |
| **R2** | Multiple pairs (3) | $0.4000$ | $0.4000$ | Unmet | FAILED |
| **R3** | Distractors (3 pairs + 2 dist) | $0.4000$ | $0.0000$ | Unmet | FAILED |
| **R4** | Reordered mappings | $0.4000$ | $0.4000$ | Unmet | FAILED |
| **R5** | Variable query positions | $0.4000$ | $0.4000$ | Unmet | FAILED |
| **R6** | Variable sequence lengths | $0.2000$ | $0.0000$ | Unmet | FAILED |
| **R7** | Disjoint identities | $0.8000$ | $0.0000$ | Unmet | FAILED |
| **R8** | Disjoint + Distractors + Shuffled | $0.6000$ | $0.8000$ | Met (stochastic) | **DIAGNOSTIC EVIDENCE** |

## Highest Systematically Passed Level
- **Level R0**: Fully mastered ($1.0000$).
- Beyond R0, multi-pair interference degrades routing consistency below the systematic $0.50$ promotion threshold across length and distractor shifts.
