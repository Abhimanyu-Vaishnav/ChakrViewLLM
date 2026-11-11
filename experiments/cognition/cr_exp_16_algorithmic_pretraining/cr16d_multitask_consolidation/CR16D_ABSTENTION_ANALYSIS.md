# CR-16D Abstention & Calibrated Answerability Analysis

### 1. Objective
Investigate whether multi-task consolidated models maintain answerability calibration by outputting `[UNKNOWN]` when queried about unbound or non-existent keys, rather than hallucinating arbitrary values from context.

### 2. Benchmark Categories Evaluated
In the CR-16C Benchmark:
- **CAT9 (Abstention on Unbound Key, 100 probes):**
  Context contains `Store: fruit = Apple.`, but query asks for `color`. Ground truth is `[UNKNOWN]`.
- **CAT10 (Abstention with Value Reassignment, 100 probes):**
  Context contains updates to other keys, query asks for unbound key. Ground truth is `[UNKNOWN]`.

---

### 3. Quantitative Abstention Results

| Model Arm | CAT9 (Unbound Key) | CAT10 (Reassignment Unbound) | Total Abstention Score |
|---|---|---|---|
| **CR-16A Arm D (Parent)** | 0.0% (0/100) | 0.0% (0/100) | 0.0% (0/200) |
| **Pure Retrieval (CR-16C)** | **58.0% (58/100)** | **20.0% (20/100)** | **39.0% (78/200)** |
| **Arm A (0% Replay, s42)** | 8.0% (8/100) | 2.0% (2/100) | 5.0% (10/200) |
| **Arm A (0% Replay, s101)** | 34.0% (34/100) | 1.0% (1/100) | 17.5% (35/200) |
| **Arm B (25% Replay, s42)** | 38.0% (38/100) | 3.0% (3/100) | 20.5% (41/200) |
| **Arm B (25% Replay, s101)** | 19.0% (19/100) | 0.0% (0/100) | 9.5% (19/200) |
| **Arm C (50% Replay, s42)** | 18.0% (18/100) | 0.0% (0/100) | 9.0% (18/200) |
| **Arm C (50% Replay, s101)** | 0.0% (0/100) | 1.0% (1/100) | 0.5% (1/200) |
| **Arm D (75% Replay, s42)** | 13.0% (13/100) | 2.0% (2/100) | 7.5% (15/200) |
| **Arm D (75% Replay, s101)** | 28.0% (28/100) | 9.0% (9/100) | 18.5% (37/200) |

---

### 4. Analysis of Abstention Dynamics
1. **Trade-off Between Retrieval Assertiveness and Abstention:**
   - In pure retrieval models (CR-16C), CAT9 abstention reached 58.0%.
   - In multi-task models with strong sequence copying (e.g. Arm C Seed 101), CAT9 drops toward 0%. When queried on an unbound key, the model attempts to retrieve *some* salient value present in context (copying bias) rather than emitting `[UNKNOWN]`.
2. **Contextual Fallback Hallucination:**
   - When the queried key does not match any stored key, the attention distribution becomes flatter across candidate value tokens in the prompt. Because the model has strong copying capability, it samples one of the visible values rather than predicting the abstention token.
3. **Implication for Calibration:**
   - Abstention is fragile under multi-task consolidation unless explicitly balanced with negative examples. Future training regimes should calibrate confidence thresholds or include dedicated null-routing targets in the training mixture.

