# Step 154 — Reasoning Curriculum Ladder

## 1. Overview

Rather than forcing the neural model across an un-scaffolded jump from zero to two-hop syllogisms, Step 154 establishes a structured 9-level progression.

## 2. Curriculum Architecture

```
[Level 0: Direct Relation Recognition]
       │ (e.g. A > B -> first: A)
       ▼
[Level 1: 1-Step Inversion & Transformation]
       │ (e.g. A > B -> last: B)
       ▼
[Level 2: 2-Hop Transitive Syllogism]
       │ (e.g. A > B and B > C -> first: A)
       ▼
[Level 3: Multi-Hop Transitive Inference]
       │ (e.g. A > B and B > C and C > D -> first: A)
       ▼
[Level 4: Distractor & Noise Robustness]
       │ (e.g. A > B and irrelevant P > Q -> first: A)
       ▼
[Level 5: Paraphrased Natural Language Relations]
       │ (e.g. A exceeds B -> first: A)
       ▼
[Level 6: Symbol & Variable Substitution]
       ▼
[Level 7: Unseen Compositions]
       ▼
[Level 8: Mixed Reasoning Families]
```

## 3. Strict Progression Criteria

Curriculum advancement is governed by:
$$\text{Held-Out Accuracy} \ge \tau \quad (\tau = 0.50)$$

Models may not advance to higher levels based merely on training set memorization or low cross-entropy loss.
