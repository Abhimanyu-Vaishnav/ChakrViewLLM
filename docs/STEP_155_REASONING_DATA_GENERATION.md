# Step 155 — Procedural Reasoning Data Generation & Contamination Control

## 1. Deterministic Procedural Generation

To prevent benchmark degradation and data exhaustion, Step 155 implements the `ProceduralReasoningGenerator` supporting:
- Ordering relations ($>$, $<$)
- Transitive relations ($A > B \land B > C \implies A > C$)
- Paraphrased templates ("exceeds", "ranks above", "is greater than")
- Multi-hop chains (3-hop, 4-hop)
- Distractor insertion
- Contradictory premises and polarity reversals

## 2. Contamination Defense Architecture

To ensure zero benchmark contamination:
1. **Semantic Normalization:** Each logical relationship is normalized to a canonical directed graph edge representation.
2. **Cryptographic Graph Hashing:** Normalized triples `(source, relation, target)` are hashed with SHA-256.
3. **Disjoint Entity Partitions:** Train entities ($A, B, C, D, E, F, M, N, P$) and held-out entities ($X, Y, Z, U, V, W, H, I, J$) are disjoint.
4. **Contamination Check Result:**
   $$\text{Contamination Rate} = 0.0000 \quad (\text{Is Clean} = \text{True})$$
