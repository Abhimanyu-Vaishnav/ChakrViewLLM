# CR-16D G3 Unseen-Key Generalization Audit

### 1. The Claim in CR-16D
CR-16D reported up to **94.4%** accuracy on Group G3 (probes with unseen keys paired with familiar values), concluding that key routing is abstract and positionally invariant.

### 2. Diagnostic Testing: Position Permutation & Conflicting Distractors
To verify whether the model relies on positional bias (e.g. always reading the first or second fact) or whether it truly searches for the key token across the context:

#### Position Permutation Test (100 matched pairs, 200 probes):
- **Orig:** `Store: key1 = val1. Store: key2 = val2. Query: key1.` $\to$ Expected: `val1`
- **Perm:** `Store: key2 = val2. Store: key1 = val1. Query: key1.` $\to$ Expected: `val1`

#### Quantitative Results:
| Model Condition | Both Orders Correct | Orig Only Correct | Perm Only Correct | Both Incorrect | Position Invariance Rate |
|---|---|---|---|---|---|
| **Arm C (50% Replay, s42)** | 0 / 100 (0.0%) | 46 / 100 (46.0%) | 0 / 100 (0.0%) | 54 / 100 (54.0%) | **0.0% (Total Failure)** |
| **Arm C (50% Replay, s101)**| 0 / 100 (0.0%) | 58 / 100 (58.0%) | 0 / 100 (0.0%) | 42 / 100 (42.0%) | **0.0% (Total Failure)** |
| **Arm B (25% Replay, s101)**| 0 / 100 (0.0%) | 46 / 100 (46.0%) | 0 / 100 (0.0%) | 54 / 100 (54.0%) | **0.0% (Total Failure)** |
| **Arm D (75% Replay, s42)** | 0 / 100 (0.0%) | 50 / 100 (50.0%) | 0 / 100 (0.0%) | 50 / 100 (50.0%) | **0.0% (Total Failure)** |

#### Conflicting Distractor Interventions (100 probes):
- Accuracy across models: **49% to 61%**.
- When an adjacent distractor fact is placed immediately before the query, models often get captured by the distractor value.

---

### 3. Mechanistic Discovery: Absolute Positional Bias
1. **The Origin of High G3 Accuracy in CR-16D:**
   - In the benchmark suite, single-fact queries (`Store: KEY = VAL. Query: KEY.\nAnswer: `) placed the target fact at an invariant token offset from the beginning of the prompt.
   - The attention head learned to route from the query position to the exact token offset of the value token. Because the query format was identical, it correctly extracted `VAL` regardless of what token was in `KEY`.
2. **Failure on Permuted Facts:**
   - When the target fact is moved from position 1 to position 2 (`Store: key2 = val2. Store: key1 = val1. Query: key1.`), **zero models succeeded on both orders (0/100)**.
   - Every single model succeeded *only* when the target was in position 1 (`Orig Only: 46% - 58%`), and failed when the target moved to position 2 (`Perm Only: 0%`).
3. **Scientific Conclusion:**
   The reported "unseen-key generalization" in CR-16D was largely an artifact of **fixed positional routing** in single-fact templates, not an invariant query-matching search over arbitrary multi-fact contexts.

