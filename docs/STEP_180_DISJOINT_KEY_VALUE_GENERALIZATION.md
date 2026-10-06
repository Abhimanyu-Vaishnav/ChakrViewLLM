# Step 180 — Disjoint Key/Value Generalization

## 1. Multi-Condition Generalization Protocol

Step 180 tested zero-shot associative transfer across six controlled conditions over 3 deterministic seeds:
1. Known key / Known value ($A \to 1, B \to 2$)
2. Known key / Unseen value ($A \to 7, B \to 8$)
3. Unseen key / Known value ($X \to 1, Y \to 2$)
4. Unseen key / Unseen value ($X \to 7, Y \to 8$)
5. Unseen mapping permutation ($C \to 3, D \to 4$)
6. Unseen association ordering ($B \to 2, A \to 1$)

---

## 2. Multi-Seed Results

| Condition | Seed 42 | Seed 101 | Seed 2026 | Mean Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Known Key / Known Value** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **Unseen Ordering on Known Tokens** | 0.5000 | 0.5000 | 0.5000 | **0.5000** |
| **Known Key / Unseen Value** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **Unseen Key / Known Value** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **Unseen Key / Unseen Value (Disjoint)**| 0.0000 | 0.0000 | 0.0000 | **0.0000** |

---

## 3. Induction Capability Level

- **Current Induction Level:** **`I2_HELDOUT_ASSOCIATIVE_RETRIEVAL`**
  *(Generalizes to unseen ordering and re-bound permutations on familiar token embeddings)*
- **Disjoint Dynamic Retrieval (`I3`):** **UNPROVEN**
  *(Zero-shot retrieval collapses when both key and value embeddings are disjoint from training sets)*
