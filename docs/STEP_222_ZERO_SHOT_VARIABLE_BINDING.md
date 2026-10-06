# Step 222: Zero-Shot Variable Binding

## 1. Scientific Objective
Investigates dynamic zero-shot variable binding where contextual mappings are completely randomized on every episode:
- Roles change per sequence (e.g. $B \to Y$ in one episode, $R \to 0$ in the next).
- Key identities, value identities, layouts, and pair counts randomized.
- Anti-shortcut controls: Balanced target frequencies, randomized distractor counts, and variable sequence lengths.
- Evaluates both:
  - Stage B: Value representation retrieval.
  - Stage C: Final token generation.

## 2. Multi-Seed Variable Binding Evaluation (Seeds 42, 101, 2026)

| Seed | Episodes | Mean Assoc Score | Mean Value Margin | Mean Value Rank | Value Rep Acc | Final Token Acc | Verified? |
|---|---|---|---|---|---|---|---|
| 42 | 20 | 0.2625 | -0.0410 | 2.65 | 0.0500 | 0.0000 | **False** |
| 101 | 20 | 0.2590 | -0.0435 | 2.70 | 0.0500 | 0.0000 | **False** |
| 2026 | 20 | 0.2610 | -0.0395 | 2.60 | 0.0500 | 0.0000 | **False** |

## 3. Scientific Distinction: Probe vs Capability
- As mandated by the Non-Negotiable Baseline:
  > "A candidate cannot be called variable-binding capable merely because a probe can decode information. Promotion requires actual contextual retrieval."
- While linear probing diagnostic tests show that key and value identities are decodable, the operational zero-shot variable binding retrieval is **0.0000**.
- Zero-shot variable binding capability remains **UNPROVEN**.
