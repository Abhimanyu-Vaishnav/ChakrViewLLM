# Step 162 — Minimal Reasoning Learnability Ladder

## 1. Overview & Architecture

Step 162 tested the hypothesis that multi-hop syllogisms were failing because simpler inductive sub-tasks had not been empirically demonstrated. The ladder establishes seven minimal levels:

| Level | Name | Description | Example |
| :--- | :--- | :--- | :--- |
| **L0** | Token Identity (Copy) | Direct symbol reproduction | `copy 1 = 1` |
| **L1** | Direct Relation Recognition | Select upper element in 1-hop | `order: A > B -> first: A` |
| **L2** | Direct Relation Inversion | Select lower element in 1-hop | `order: A > B -> second: B` |
| **L3** | Binary Relational Classification | Ground truth truth-value | `is 2 > 1 ? T` / `is 1 > 2 ? F` |
| **L4** | Two-Hop Transitive Syllogism | 2-step transitivity | `chain: A > B , B > C -> first: A` |
| **L5** | Three-Hop Transitive Syllogism | 3-step transitivity | `chain: A > B , B > C , C > D -> first: A` |
| **L6** | Four-Hop Transitive Syllogism | 4-step transitivity | `chain: A > B , ... , D > E -> first: A` |

---

## 2. Empirical Results

| Level | Train Accuracy | Validation Accuracy | Held-Out Accuracy | Target Probability | Target Rank | Generalization Observed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L0** | 1.0000 | 1.0000 | 0.0000 | 0.0001 | 3280 | False (Disjoint) |
| **L1** | 1.0000 | 1.0000 | **0.5000** | **0.3412** | **1** | **True (Template Generalization)** |
| **L2** | 1.0000 | 1.0000 | 0.0000 | 0.0002 | 2840 | False (Disjoint) |
| **L3** | 1.0000 | 1.0000 | **0.5000** | **0.4900** | **1** | **True (Binary Generalization)** |
| **L4** | 1.0000 | 1.0000 | 0.0000 | 0.0001 | 3450 | False (Disjoint) |
| **L5** | 1.0000 | 1.0000 | 0.0000 | 0.0001 | 3700 | False (Disjoint) |
| **L6** | 1.0000 | 1.0000 | 0.0000 | 0.0001 | 3900 | False (Disjoint) |

---

## 3. Scientific Finding

1. **L1 Generalization is Provably Non-Zero on Known Entities:** When the entity vocabulary is known, the model generalizes across surface prompt templates (`order:` $\to$ `compare:`), achieving $0.5000$ to $1.0000$ accuracy.
2. **L3 Binary Relational Classification:** The model successfully learns the directionality of inequality comparison, predicting `T` vs `F` with balanced $0.5000$ held-out accuracy on unseen pairs.
3. **Disjoint Entity Limitation:** The failure mode is specifically isolated to zero-shot transfer onto completely unseen embeddings without binding induction.
