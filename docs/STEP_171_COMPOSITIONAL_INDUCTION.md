# Step 171 — Compositional Induction Curriculum

## 1. Compositional Ladder Architecture

Step 171 constructed an 8-level compositional induction ladder:
- **L0:** Token Identity / Direct Copying (`copy 1 = 1`)
- **L1:** Direct Relation (1-hop ordering: `order: A > B -> first: A`)
- **L2:** Relation Inversion (`order: A > B -> second: B`)
- **L3:** Two-Hop Composition (`chain: A > B , B > C -> first: A`)
- **L4:** Three-Hop Composition (`chain: A > B , B > C , C > D -> first: A`)
- **L5:** Four-Hop Composition (`chain: A > B , ... , D > E -> first: A`)
- **L6:** Distractor-Aware Reasoning (`fact: A > B , noise: 7 > 8 -> first: A`)
- **L7:** Unseen Compositions (`query: A > B and B > C -> min: C`)

---

## 2. Contamination Defense Audit

Every relational graph premise was converted to an invariant canonical representation and cryptographically hashed using SHA-256:
- **Total Training Hashes:** 20
- **Total Held-Out Hashes:** 16
- **Leaked Hashes:** 0
- **Measured Contamination Rate:** $\mathbf{0.0000}$ (`is_clean = True`).
