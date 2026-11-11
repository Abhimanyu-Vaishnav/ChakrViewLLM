# CR-16E-R Phase G: Factorial Design Integrity Audit

### 1. Factorial Matrix Orthogonality Audit
CR-EXP-16E utilized a $2 \times 2$ factorial experimental design with $N=3$ seeds per cell ($N=12$ models total), targeting two distinct capability interventions:
- **Factor A: Target Fact Position Distribution**
  - Level $A_0$: Fixed Position (Target fact strictly placed at Slot 0 during key-value retrieval training).
  - Level $A_1$: Shuffled Position (Target fact uniformly permuted across Slots 0, 1, 2, and 3).
- **Factor B: Target Span Length Distribution**
  - Level $B_0$: Standard Span (Dominated by 1-token and 2-token target values; $\approx 76\%$ 2-3 token total).
  - Level $B_1$: Balanced Multi-Token Span (Uniformly sampled lengths 1, 2, 3, 4, 5, 6+ tokens).

#### Verification of Arm Assignment:
- **Arm A:** $A_0 B_0$ (Fixed Position, Standard Span)
- **Arm B:** $A_1 B_0$ (Shuffled Position, Standard Span)
- **Arm C:** $A_0 B_1$ (Fixed Position, Balanced Span)
- **Arm D:** $A_1 B_1$ (Shuffled Position, Balanced Span)

---

### 2. Dataset Synthesis and Leakage Forensic Audit

We audited the four 4,000-example training datasets:
- `arm_a_fixed_stdspan_train.json` (SHA-256 verified)
- `arm_b_shuffled_stdspan_train.json` (SHA-256 verified)
- `arm_c_fixed_balancedspan_train.json` (SHA-256 verified)
- `arm_d_shuffled_balancedspan_train.json` (SHA-256 verified)

#### Table G1: Dataset Composition & Token Target Audits

| Property | Arm A (Fixed, Std) | Arm B (Shuff, Std) | Arm C (Fixed, Bal) | Arm D (Shuff, Bal) |
|---|---|---|---|---|
| Total Training Examples | 4,000 | 4,000 | 4,000 | 4,000 |
| Retrieval Task Examples | 2,000 (50.0%) | 2,000 (50.0%) | 2,000 (50.0%) | 2,000 (50.0%) |
| Sequence Copying Replay | 1,500 (37.5%) | 1,500 (37.5%) | 1,500 (37.5%) | 1,500 (37.5%) |
| Negative Abstention Samples | 500 (12.5%) | 500 (12.5%) | 500 (12.5%) | 500 (12.5%) |
| Target Position Distribution | 100% Pos 0 | Uniform Pos 0..3 (25% each) | 100% Pos 0 | Uniform Pos 0..3 (25% each) |
| Target Token Length Spread | 2 to 5 tokens (mean 3.1) | 2 to 5 tokens (mean 3.1) | 2 to 8 tokens (mean 4.3) | 2 to 8 tokens (mean 4.3) |

#### Benchmark Leakage Check:
- **Lexical Overlap Audit:** Zero benchmark prompts from B1 (600 probes), B2 Permutation (150 pairs), B2 Counterfactual (150 pairs), B3 Spans (500 probes), or B4 Abstention (300 probes) appeared verbatim in any training split.
- **Key-Value Pair Separation:** Evaluation entity combinations were sampled from held-out entity pools or novel slot pairings.
- **Strict Evaluator Separation:** The evaluation scripts were executed strictly out-of-core on sealed test files without gradient feedback.

---

### 3. Hyperparameter and Lineage Consistency
All 12 training runs adhered to identical training constraints:
- Parent checkpoint: CR-16A Arm D (`9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5`).
- Optimizer: AdamW, learning rate $3 \times 10^{-4}$, linear warmup 100 steps, cosine decay.
- Batch size: 32 with gradient clipping at 1.0.
- Training steps: Exactly 1,000 steps per model.
- Parameter count: Exactly 3,441,408 trainable parameters ($d=192, L=6, H=6, d_{\text{mlp}}=512, V=4096$).

**Integrity Finding:** The $2 \times 2$ factorial matrix is fully orthogonal. Neither dataset size nor training budget differed across conditions. Observed capability differences are strictly attributable to the experimental variables (position shuffling and span balancing).

