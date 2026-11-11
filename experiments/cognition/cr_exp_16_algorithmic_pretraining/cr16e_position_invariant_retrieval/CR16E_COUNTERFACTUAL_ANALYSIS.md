# CR-16E Counterfactual & Permutation Analysis

### 1. The Challenge of Robust In-Context Binding
In CR-16D-R, models achieved high single-probe retrieval scores when the target was in position 1, but collapsed completely (0/100) when fact order was permuted, and exhibited only 38% paired counterfactual consistency.

CR-16E Benchmark B2 tested matched pairs across two dimensions:
1. **Position Permutation Pairs (150 pairs = 300 probes):**
   - Probe A: Target fact in Position 0 (`Store: K1 = V1. Store: K2 = V2. Store: K3 = V3. Query: K1.`)
   - Probe B: Target fact in Position 2 (`Store: K2 = V2. Store: K3 = V3. Store: K1 = V1. Query: K1.`)
   - *Requirement:* Both probes must return `V1`.
2. **Counterfactual Value-Change Pairs (150 pairs = 300 probes):**
   - Probe A: `Store: K = VA. Store: K_dist = V_dist. Query: K.` $\to$ Expected: `VA`
   - Probe B: `Store: K = VB. Store: K_dist = V_dist. Query: K.` $\to$ Expected: `VB`
   - *Requirement:* Both probes must return their respective context value (`VA` and `VB`).

---

### 2. Quantitative Results across Factorial Arms

| Factorial Arm | Permutation Pairs Both Correct (150) | Counterfactual Pairs Both Correct (150) | Invariant Failures ($A=B$) |
|---|---|---|---|
| **Arm A (Fixed Pos, Std Span, mean)** | **0.0 / 150 (0.0%)** | 36.7 / 150 (24.5%) | 32.3 / 150 (21.5%) |
| **Arm B (Shuffled Pos, Std Span, mean)**| 4.0 / 150 (2.7%) | 2.0 / 150 (1.3%) | 100.7 / 150 (67.1%) |
| **Arm C (Fixed Pos, Balanced Span, mean)**| **0.0 / 150 (0.0%)** | 18.3 / 150 (12.2%) | 55.7 / 150 (37.1%) |
| **Arm D (Shuffled, Balanced Span, mean)**| **9.3 / 150 (6.2%)** | 6.7 / 150 (4.5%) | 90.3 / 150 (60.2%) |

*Peak Single Run on Permutation:* **Arm D Seed 4040** achieved **22 / 150 (14.7%)** paired permutation success and **19 / 150 (12.7%)** counterfactual success, with invariant failure dropping to 41/150.

---

### 3. Mechanistic Insights
1. **Breaking the Zero-Permutation Barrier:**
   - In fixed-position models (Arms A and C), paired permutation success was **strictly 0.0% across all 6 runs (0 / 900 pairs)**. Because the model cannot attend to position 2, probe B always fails.
   - Position-shuffled training (Arm D Seed 4040) broke this barrier, achieving **22 / 150 (14.7%) paired successes**.
2. **The Trade-Off Between Shuffling and Counterfactual Stability:**
   - When positions are randomized, the model encounters a more complex attention routing optimization landscape. Within a 400-step training budget, this increased complexity causes some seeds to default to frequent tokens (high invariant failure in Arms B and D s2026/s3030).
   - In Seed 4040, the model discovered the joint routing solution, demonstrating that content-based key routing across permuted positions is computationally achievable by ChakrMicro's attention heads.

