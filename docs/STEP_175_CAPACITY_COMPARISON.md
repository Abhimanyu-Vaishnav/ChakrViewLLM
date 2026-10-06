# Step 175 — Controlled Model Capacity Comparison

## 1. Experimental Methodology

Step 175 conducted an empirical size scaling study comparing the canonical 3.44M ChakrMicro model against an experimental 9.15M model under strictly identical conditions:
- **Same Tokenizer:** BPE 4096 vocabulary
- **Same Optimizer:** AdamW, learning rate $1\times 10^{-3}$, 30 optimization steps
- **Same Dataset & Seeds:** Identical prompt pairs and random initialization

---

## 2. Quantitative Comparison Matrix

| Model Architecture | Parameter Count | Train Acc | Held-Out Template Acc (G2) | Held-Out Disjoint Acc (G1) | Final Loss | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ChakrMicro (Canonical)** | 3,443,136 | 1.0000 | 0.5000 | 0.0000 | 0.9421 | 1.82s |
| **ChakrSmall (Experimental)**| 9,147,168 | 1.0000 | 0.5000 | 0.0000 | 0.6812 | 3.45s |

---

## 3. Scientific Finding

1. Scaling from 3.44M to 9.15M parameters accelerates training convergence (final loss $0.94 \to 0.68$).
2. However, scaling parameter count **DOES NOT** automatically produce zero-shot disjoint entity transfer ($0.0000 \to 0.0000$).
3. **Classification:** **`CAPACITY_ALONE_INSUFFICIENT`**.
   The bottleneck cannot be solved simply by increasing model size; it requires structural variable-binding induction pretraining.
