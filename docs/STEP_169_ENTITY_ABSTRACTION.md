# Step 169 — Entity Abstraction & Token Representation Experiment

## 1. Experimental Methodology & Splits

Step 169 investigated whether ChakrMicro learns relational roles independently of the surface token identities of participating entities.
Four structured experimental conditions were tested:
1. **Known Entities / Known Template:** Train set ($A > B \to A$)
2. **Known Entities / Unseen Template:** Paraphrased syntax on known symbols ($A > B$ under `compare:`)
3. **Unseen Entities / Known Template:** Disjoint entity symbols ($X > Y \to X$) under familiar `order:` syntax
4. **Unseen Entities / Unseen Template:** Disjoint entity symbols under unseen `compare:` syntax

Evaluated across symbol sets:
- Uppercase ASCII letters ($A, B, C, \dots$)
- Lowercase ASCII letters ($a, b, c, \dots$)
- Digits ($1, 2, 3, \dots$)

---

## 2. Multi-Seed Empirical Results

Tested across three deterministic seeds (**42, 101, 2026**) with verified single-byte token encodings:

| Condition | Seed 42 Acc | Seed 101 Acc | Seed 2026 Acc | Mean Accuracy | Median Target Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Known Entities / Known Template** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1 |
| **Known Entities / Unseen Template** | 0.2500 | 0.2500 | 0.2500 | **0.2500** | 1 |
| **Unseen Entities / Known Template** | 0.0000 | 0.0000 | 0.0000 | **0.0000** | > 2500 |
| **Unseen Entities / Unseen Template** | 0.0000 | 0.0000 | 0.0000 | **0.0000** | > 3000 |

---

## 3. Core Scientific Finding

**Question:** *Does the model learn entity ROLE independently of entity ID?*
**Answer:** **NO.**
The model associates relational roles strictly with bound embeddings of familiar training entities. When evaluated on unseen disjoint tokens, the attention heads and output projections fail to copy the role abstractly, resulting in an immediate rank drop to $> 2500$.
