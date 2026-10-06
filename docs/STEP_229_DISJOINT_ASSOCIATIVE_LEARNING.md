# Step 229: Disjoint Associative Generalization

## 1. Scientific Objective
Evaluate the Primary **I3 Promotion Gate**:
- Training Set: Keys $\{A, B, C, D, E\}$, Values $\{1, 2, 3, 4, 5\}$.
- Evaluation Set: Keys $\{P, Q, R, S, T\}$, Values $\{6, 7, 8, 9, 0\}$.
- Mappings randomized per episode. No fixed keys, values, or answer positions.

## 2. Multi-Seed Split Performance

Evaluated across seeds 42, 101, and 2026:

| Seed | Condition | Assoc Score | Value Rep Margin | Final Token Prob | Final Token Acc | Qualified for I3? |
|---|---|---|---|---|---|---|
| 42 | `known_known` | 0.2810 | -0.0002 | 0.4210 | 0.6000 | In-Dist Only |
| 42 | `known_unseen` | 0.2640 | -0.0150 | 0.0012 | 0.0000 | No |
| 42 | `unseen_known` | 0.2710 | -0.0120 | 0.1200 | 0.2000 | Partial Transfer |
| 42 | `unseen_unseen` | 0.2580 | -0.0380 | 0.0010 | 0.0000 | **No** |
| 101 | `known_known` | 0.2790 | -0.0005 | 0.4150 | 0.6000 | In-Dist Only |
| 101 | `unseen_unseen` | 0.2560 | -0.0395 | 0.0009 | 0.0000 | **No** |
| 2026 | `known_known` | 0.2830 | -0.0001 | 0.4280 | 0.6000 | In-Dist Only |
| 2026 | `unseen_unseen` | 0.2600 | -0.0370 | 0.0011 | 0.0000 | **No** |

## 3. Promotion Rule Evaluation
- I3 requires:
  1. Disjoint keys & disjoint values evaluated.
  2. Randomized mappings per episode.
  3. Meaningful improvement on unseen-unseen condition ($\text{Acc} \ge 0.50$).
  4. Multiple seeds evaluated separately.
- **Empirical Status**:
  - `unseen_unseen` final token accuracy is **0.0000** across all three seeds.
- **Verdict**:
  $$\mathbf{I3\_PROMOTION\_DENIED\_ZERO\_DISJOINT\_RETRIEVAL}$$
  The candidate fails the I3 gate. Capability remains at **`I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`**.
