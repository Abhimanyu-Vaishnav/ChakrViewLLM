# Step 219: Disjoint Value Representation Transfer

## 1. Scientific Objective
Investigates whether the association mechanism can retrieve a **VALUE REPRESENTATION** for an unseen entity even when final vocabulary output fails:
- Train Pool: Keys $\{A, B, C, D, E\}$, Values $\{1, 2, 3, 4, 5\}$
- Test Pool: Keys $\{P, Q, R, S, T\}$, Values $\{6, 7, 8, 9, 0\}$
- Mappings randomized per episode.

## 2. Multi-Seed Transfer Results (Seeds 42, 101, 2026)

| Seed | Condition | Assoc Score | Value Rep Cosine | Value Rep Margin | Value Rep Rank | Final Token Prob | Final Token Acc | Rep Transfer? | Token Transfer? |
|---|---|---|---|---|---|---|---|---|---|
| 42 | `known_known` | 0.2840 | 0.4120 | -0.0365 | 2.50 | 0.0012 | 0.0000 | False | False |
| 42 | `unseen_unseen` | 0.2601 | 0.3890 | -0.0460 | 2.70 | 0.0009 | 0.0000 | False | False |
| 101 | `known_known` | 0.2815 | 0.4105 | -0.0350 | 2.45 | 0.0013 | 0.0000 | False | False |
| 101 | `unseen_unseen` | 0.2590 | 0.3875 | -0.0480 | 2.75 | 0.0008 | 0.0000 | False | False |
| 2026 | `known_known` | 0.2830 | 0.4110 | -0.0360 | 2.50 | 0.0011 | 0.0000 | False | False |
| 2026 | `unseen_unseen` | 0.2610 | 0.3900 | -0.0450 | 2.65 | 0.0009 | 0.0000 | False | False |

## 3. Investigation Target Determination
- Criteria:
  - If representation retrieval succeeds while token output fails: Target = `VOCABULARY_PROJECTION`.
  - If representation retrieval also fails: Target = `ASSOCIATIVE_ROUTING`.
- **Verdict**:
  $$\text{Value Representation Retrieval Failed} \implies \text{Next Investigation Target} = \mathbf{ASSOCIATIVE\_ROUTING}$$
- Disjoint value representation transfer is **REFUTED** on the unadapted backbone.
