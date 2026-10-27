# EXPERIMENT 32 PROTOCOL
## Execution Steps, Data Deduplication Guarantees & Error Diagnostics

### 1. Data Integrity & Leakage Elimination Protocol
1. Initialize a global prompt tracking registry: `seen_prompts = set()`.
2. Any candidate record whose `prompt` already exists in `seen_prompts` must be rejected immediately during generation.
3. Every test group must be verified to have $Split \cap seen\_train\_prompts = \emptyset$.
4. All single-token verifications for $V_{train}$, $V_{unseen}$, and `'?'` must be asserted against `load_tokenizer_artifacts`.

### 2. Diagnostic Metrics & Failure Categories
For every evaluation instance across all groups, record:
- `target`: Expected answer token.
- `pred_text`: Model predicted string.
- `pred_id`: Predicted token ID.
- `is_correct`: Exact-match boolean.
- `is_abstain`: True if `pred_text == '?'`.
- `failure_type`:
  - `CORRECT`: Prediction matches target.
  - `MISSED_ABSTENTION`: Target was `?`, but model predicted a context token.
  - `FALSE_ABSTENTION`: Target was a valid token, but model predicted `?`.
  - `DISTRACTOR_SELECTION`: Model predicted a value from context that was bound to a non-queried key.
  - `HALLUCINATION_OR_BIAS`: Model predicted a token that does not exist in context.

### 3. Step-by-Step Execution Sequence
1. Run `generate_dataset.py` with seed `42` under `experiments/exp32_pointer_abstention/`.
2. Compute and log the SHA-256 hash of `PREREGISTRATION.md`.
3. Verify zero cross-split prompt leakage across all splits.
4. Execute `run_exp32.py` evaluating Arms 0, 1, and 2.
5. Compute per-group exact matches, abstention precision/recall, and distractor error frequencies.
6. Run full regression test suite (`tests/test_v060_release_smoke.py`).
7. Verify bit-exact hashes of canonical weights, tokenizer, and protected files.
8. Generate artifact manifest `MANIFEST.json`.
9. Author `EXP32_REPORT.md`.
