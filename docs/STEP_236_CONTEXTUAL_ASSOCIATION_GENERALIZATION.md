# Step 236: Contextual Association Generalization

## 1. Scientific Objective
Evaluate associative learning under systematic contextual and surface variations where semantic operations remain constant:
- Alternating separator syntax (` ; ` vs. ` and `)
- Varying sequence lengths (2, 3, 4 associations)
- Shuffled presentation orders
- Evaluated across the 4 canonical generalization splits:
  1. `known_known`: Familiar keys, familiar values
  2. `known_unseen`: Familiar keys, unseen values
  3. `unseen_known`: Unseen keys, familiar values
  4. `unseen_unseen`: Disjoint unseen keys, disjoint unseen values

## 2. Generalization Split Evaluation (Seed 42)

| Split | Token Accuracy | Value Rep Margin | Value Rep Rank | Assoc Score | Token Probability | Target Status |
|---|---|---|---|---|---|---|
| `known_known` | 0.0000 | -0.0365 | 2.50 | 0.2840 | 0.0012 | Zero-shot baseline |
| `known_unseen` | 0.0000 | -0.0370 | 2.60 | 0.2715 | 0.0010 | Zero-shot baseline |
| `unseen_known` | 0.0000 | -0.0330 | 2.50 | 0.2650 | 0.0011 | Zero-shot baseline |
| `unseen_unseen` | 0.0000 | -0.0460 | 2.70 | 0.2601 | 0.0009 | **Target > 0.0: Not Met** |

## 3. Scientific Verdict
- In the unadapted baseline, zero-shot token generation on unseen entities remains $0.0000$ across all tested syntactic variations.
- Without associative circuit adaptation, surface prompt variations alone cannot elicit out-of-distribution associative routing.
