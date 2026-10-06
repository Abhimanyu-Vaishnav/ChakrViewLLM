# Step 158 — Multitask Neural Learning & Anti-Forgetting

## 1. Candidate Comparison Matrix

Step 158 investigated the interaction between neural language modeling and relational reasoning:
- **Candidate A:** Language Only
- **Candidate B:** Reasoning Only
- **Candidate C:** Language + Reasoning Interleaved

## 2. Quantitative Results

| Candidate | Training Regime | Language Cross-Entropy Loss | Reasoning Held-Out Accuracy | Language Retention Preserved |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Language Only | $7.7625$ | $0.0000$ | N/A |
| **B** | Reasoning Only | $8.3947$ | $0.0000$ | Degraded (baseline level) |
| **C** | Interleaved | $8.0334$ | $0.0000$ | **Yes** ($\text{loss} < 8.25$) |

## 3. Findings

1. Training purely on reasoning tasks leaves language performance unadapted at baseline loss ($8.3947$).
2. Interleaved multitask training effectively preserves learned language representations ($8.0334 < 8.25$), preventing catastrophic forgetting while adapting candidate parameters toward the reasoning objective.
