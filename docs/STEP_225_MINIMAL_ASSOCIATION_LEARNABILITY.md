# Step 225: Minimal Association Learnability

## 1. Scientific Objective
Determine whether an isolated ChakrMicro candidate can learn the smallest contextual association task through controlled CPU gradient training without fixed-mapping memorization.

Prompt structure:
`map |A| -> |X| and |B| -> |Y| and |C| -> |Z| query |B| -> |`
Mappings randomized independently on every training and validation episode.

## 2. Training & Evaluation Protocol
- Model: Isolated clone of canonical frozen ChakrMicro (3,443,136 parameters).
- Optimizer: AdamW, learning rate $3 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, gradient clipping norm 1.0.
- Hardware: 100% CPU PyTorch execution.
- Baseline check: Bit-exact SHA-256 confirmed before and after run.
- Metrics measured:
  - Training association accuracy
  - Validation association accuracy
  - Held-out mapping accuracy
  - Association representation score
  - Value representation retrieval margin & rank
  - Final token accuracy
  - Language retention (validation loss delta)
  - Parameter delta norm ($\|\theta_{\text{cand}} - \theta_{\text{base}}\|$)

## 3. Empirical Results (Seed 42)

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Training Loss | 1.1082 | N/A | Converging |
| Mean Gradient Norm | 0.8845 | $\le 1.0$ | Stable |
| Train Association Accuracy | 0.6000 | $\ge 0.80$ | Partial Learning |
| Validation Association Accuracy | 0.6000 | $\ge 0.70$ | Partial Generalization |
| Held-Out Mapping Accuracy | 0.0000 | $\ge 0.50$ | Unproven |
| Association Representation Score | 0.0000 | $\ge 0.50$ | Weak Contextual Separation |
| Value Representation Margin | -0.0002 | $> 0.0$ | Negative Margin |
| Value Representation Rank | 1.90 | $< 2.0$ | Above Chance |
| Language Loss (Before $\to$ After) | 7.7396 $\to$ 8.3461 | $\le 1.25\times$ | Preserved (7.8% drift) |
| Parameter Delta Norm | 0.8924 | N/A | Measurable Gradient Update |

## 4. Scientific Verdict
- **Candidate Acceptance State**: `HOLD_FOR_INVESTIGATION`.
- Controlled neural learning produces in-distribution associative retrieval ($0.00 \to 0.60$), but does not reach the minimal promotion threshold ($0.80$ train, $0.70$ val) on random episode mappings, and held-out transfer remains $0.0000$.
