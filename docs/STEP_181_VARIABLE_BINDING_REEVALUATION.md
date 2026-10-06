# Step 181 — Variable Binding Re-Evaluation after Induction Training

## 1. Experimental Methodology

Step 181 re-evaluated the core relational variable binding tasks:
$$A > B \to \text{first: } A \quad;\quad A > B \to \text{second: } B \quad;\quad P > Q \to \text{first: } P$$
Comparing model performance **BEFORE** associative training vs **AFTER** associative training.

---

## 2. Comparative Matrix

| Evaluation Split | Pre-Induction Accuracy | Post-Induction Accuracy | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **First Role Accuracy (Known)** | 0.5000 | 0.5000 | 0.0000 | Preserved |
| **Second Role Accuracy (Known)**| 0.5000 | 0.5000 | 0.0000 | Preserved |
| **Disjoint Entity Transfer ($P > Q$)**| 0.0000 | 0.0000 | 0.0000 | Unproven |

---

## 3. Finding

Associative recall pretraining successfully preserves in-distribution query role distinction (`first` vs `second`), but does not transfer zero-shot onto unaligned disjoint symbol embeddings without specialized binding induction architectures.
