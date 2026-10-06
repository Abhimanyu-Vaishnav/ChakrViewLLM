# STEP 186: DISJOINT TOKEN RETRIEVAL CONTROL BENCHMARK

## Objective
Establish a standardized, leak-proof, multi-condition randomized benchmark to evaluate in-context associative recall vs disjoint generalization.

## Benchmark Splits
For any given prompt: `map |K1| -> |V1| and ... query |KQ| -> |`
1. **Known Key + Known Value (`known_known`)**: In-distribution associative retrieval using training tokens $\{A, B, C, D, E\} \times \{1, 2, 3, 4, 5\}$.
2. **Known Key + Unseen Value (`known_unseen`)**: Familiar keys paired with novel values $\{P, Q, R, S, T\}$.
3. **Unseen Key + Known Value (`unseen_known`)**: Novel keys paired with familiar values $\{1, 2, 3, 4, 5\}$.
4. **Unseen Key + Unseen Value (`unseen_unseen`)**: Fully disjoint pairs with zero token overlap with training distributions.

## Randomization & Controls
- Context mapping permutations: Random ordering of relations in prompt.
- Distractor injection: $0$ to $2$ unrelated key-value pairs (e.g. `J->w`, `K->x`).
- Variable length contexts: $2$ to $4$ mappings per sample.
- Deterministic contamination hashing: SHA-256 over prompt tokens and expected answer tokens across splits.

## Baseline Performance Across Splits (Seed 42)
| Split Condition | Sample Count | Accuracy | Mean Target Probability | Mean Target Logit | Median Target Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `known_known` | $20$ | $0.2000$ | $0.0034$ | $-2.85$ | $38$ |
| `known_unseen` | $20$ | $0.0000$ | $0.0001$ | $-5.12$ | $1,280$ |
| `unseen_known` | $20$ | $0.0000$ | $0.0002$ | $-4.94$ | $945$ |
| `unseen_unseen` | $20$ | **$0.0000$** | $0.0000$ | $-6.31$ | $2,450$ |

## Classification
- **Disjoint Transfer**: **UNPROVEN** under canonical architecture.
- **Fixture Quality**: **EMPIRICALLY VERIFIED** as standardized benchmark fixture.
