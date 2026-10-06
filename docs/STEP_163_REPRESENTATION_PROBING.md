# Step 163 — Representation Probing Diagnostics

## 1. Diagnostic Purpose & Protocol

Step 163 investigated whether internal hidden representations contain relational information (relational direction, ordering, and entity syntax) even when the final token prediction is imperfect.
- **Rule:** Diagnostic probes are trained strictly outside the model on frozen hidden activations.
- **Rule:** Probes are NEVER counted as neural reasoning capability; they serve exclusively as diagnostic evidence.

---

## 2. Layer-Wise Probe Results

Target Property: **Relational Direction & Syntax Inversion** (`A > B` vs `B < A`).

| Layer Index | Baseline Probe Accuracy | Candidate Probe Accuracy | Separability Score | Relational Signal Present |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 0** | 0.8750 | **1.0000** | **0.9747** | **Yes** |
| **Layer 1** | 0.8750 | **1.0000** | **0.9612** | **Yes** |
| **Layer 2** | 0.7500 | **1.0000** | **0.9540** | **Yes** |
| **Layer 3** | 0.6250 | **1.0000** | **0.9488** | **Yes** |
| **Layer 4** | 0.6250 | **0.8750** | **0.9230** | **Yes** |
| **Layer 5** | 0.6250 | **0.8750** | **0.9105** | **Yes** |

---

## 3. Diagnostic Conclusion

- Internal representations develop linearly separable clusters distinguishing forward relations (`>`) from inverted relations (`<`) across all 6 transformer layers.
- **The Information-Readout Gap:** Relational information is present in hidden states, but the tied final projection layer requires exact embedding matching to output the correct token index at generation time.
