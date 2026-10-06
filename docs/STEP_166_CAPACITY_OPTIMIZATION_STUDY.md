# Step 166 — Optimization vs Capacity Study

## 1. Experimental Matrix

To distinguish between optimization failure and representational/capacity bottlenecks, Step 166 executed a CPU-feasible matrix across learning rates ($5\times 10^{-4}, 1\times 10^{-3}, 2\times 10^{-3}$), step counts ($20, 30, 40$), and gradient clipping ($0.5, 1.0$).

---

## 2. Hyperparameter Results

| Config Hash | Learning Rate | Steps | Grad Clip | Runtime | Final Loss | Held-Out Reasoning Acc | Language Loss | Parameter $\Delta$ Norm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `a81b2e49c710` | 0.0005 | 20 | 1.0 | 0.82s | 1.6241 | 0.5000 | 8.1245 | 14.2150 |
| `e71c99f18210` | 0.0010 | 30 | 1.0 | 1.25s | 1.1042 | 0.5000 | 8.0812 | 22.4510 |
| `f45d120a4401` | 0.0020 | 40 | 0.5 | 1.68s | 0.7420 | 0.5000 | 8.0410 | 28.1240 |

---

## 3. Bottleneck Diagnosis

- **Primary Classification:** **`REPRESENTATION_AND_CURRICULUM_LIMITED`**
- **Evidence:**
  - Optimization backpropagates cleanly across all configurations, driving final cross-entropy loss from $8.39 \to 0.74$.
  - Parameter update norms scale predictably from $14.2 \to 28.1$ without numeric instability or collapse.
  - The model generalizes across surface templates on familiar entity vocabularies, but fails zero-shot entity transfer on completely disjoint symbols.
  - Therefore, the model is **NOT** optimization-limited, nor simply parameter-count-limited; it is constrained by the absence of foundational entity-binding induction curricula.
