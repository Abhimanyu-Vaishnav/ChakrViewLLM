# EXPERIMENT 33 PROTOCOL
## Execution Sequence, Relational Diagnostics & Distractor Tracking

### 1. Data Generation & Integrity Check
- Generate data using `experiments/exp33_relational_pointer/generate_dataset.py` with seed `42`.
- Record preregistration SHA-256 before training execution.
- Programmatically verify zero prompt overlap ($Split_i \cap Split_j = \emptyset, \forall i \neq j$).

### 2. Error Categorization Schema
For every evaluation instance across all 6 groups, classify predictions into:
1. `CORRECT`: Exact match with target.
2. `DISTRACTOR_SELECTION`: Target is a valid token, but model predicted a value bound to an unqueried context key.
3. `MISSED_ABSTENTION`: Target is `?`, but model predicted an in-context value.
4. `FALSE_ABSTENTION`: Target is a valid token, but model predicted `?`.
5. `HALLUCINATION_OR_BIAS`: Model predicted a token that does not exist in context.

### 3. Execution Pipeline
1. Run `generate_dataset.py` to create frozen splits.
2. Log SHA-256 manifest of generated data.
3. Execute `run_exp33.py`:
   - Arm 0: Canonical Zero-Shot Baseline.
   - Arm 1: Standard Single-Head Pointer Baseline.
   - Arm 2: Relational Pointer Circuit (`ChakrRelationalPointerMicro`).
4. Recompute per-group metrics, distractor error frequencies, and abstention F1.
5. Run release smoke regression suite (`test_v060_release_smoke.py`).
6. Verify canonical and protected file hashes.
7. Generate artifact manifest `MANIFEST.json`.
8. Write comprehensive report `EXP33_REPORT.md`.
