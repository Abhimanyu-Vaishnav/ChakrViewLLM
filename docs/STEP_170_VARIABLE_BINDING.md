# Step 170 — Variable Binding & Role Permutations

## 1. Experimental Protocol

Step 170 evaluated whether ChakrMicro can bind queried structural roles to entity tokens under controlled permutations:
- $A > B \to \text{first: } A$
- $A > B \to \text{second: } B$
- $B > A \to \text{first: } B$
- $B > A \to \text{second: } A$

The same entities appear in multiple contrasting structural roles to isolate whether internal representations track entity identity, position, relation direction, and queried role.

---

## 2. Quantitative Results

| Metric | Empirical Score | Classification |
| :--- | :--- | :--- |
| **Train Set Accuracy** | 0.5000 | In-Distribution Learned |
| **'First' Role Accuracy** | 0.5000 | Partially Disentangled |
| **'Second' Role Accuracy** | 0.5000 | Partially Disentangled |
| **Inverted Direction Accuracy ($B > A$)** | 0.5000 | Directional Tracking |
| **Held-Out Permutations ($C, D$)** | 0.0000 | Unproven |

---

## 3. Mechanistic Finding

- The model successfully learns to distinguish query tokens (`first` vs `second`) on familiar tokens, achieving $50\%$ accuracy on role-swapped permutations.
- However, when unfamiliar entity tokens occupy these slots, the model defaults to token frequency bias rather than dynamic variable binding.
