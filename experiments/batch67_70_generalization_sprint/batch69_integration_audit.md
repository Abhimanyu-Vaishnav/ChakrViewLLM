# Batch 69 — Integration & Contract Hardening Audit

## 1. Audit Scope & Objective

The engineering mission of Batch 69 is to harden the interface contracts across data, tokenization, training, checkpointing, inference, and evaluation to prevent silent divergences (such as the newline stripping mismatch uncovered in Batch 63).

We audited:
1. **Shared Prompt/Response Formatting**: Eliminating divergent ad-hoc `.strip()` calls across different modules.
2. **Tokenizer Selection & Boundary Invariants**: Verifying that special tokens (`<|im_start|>`, `<|im_end|>`, `\n`) map to deterministic IDs.
3. **Causal Target Alignment**: Ensuring target masking consistently aligns input at position $|P|$ to target token $R_0$.
4. **Weighted Target Loss Module**: Abstracting loss computation to support baseline, answer-emphasis, and diagnostic loss regimes.
5. **Exact Match Scoring Parity**: Standardizing evaluation output parsing to prevent cosmetic differences from skewing capability metrics.

---

## 2. Hardened Architecture: Canonical Prompt Contract

Previously, `SFTDataset` (`chakrview/training/sft_engine.py`) and `DeterministicEvaluator` (`experiments/batch42_reproducible_evaluation/evaluation/evaluator.py`) duplicated custom prompt/response normalization logic.

In Batch 69, we created a single canonical helper module:
`chakrview/training/prompt_contract.py`

### Key Contract Functions:
1. `normalize_prompt_response_strings(prompt, response, strip_delimiters=False)`:
   - When `strip_delimiters=False` (Canonical): guarantees terminal delimiters (`\n` after `<|im_start|>assistant\n` and `<|im_end|>\n`) are preserved, ensuring bit-exact alignment between training conditions and autoregressive greedy decoding.
   - When `strip_delimiters=True` (Legacy): applies `.strip()` to maintain backward compatibility with Batches 42–62 test suites.
2. `build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)`:
   - Returns `input_seq = [BOS] + p_toks + r_toks` and `target_seq = [ignore_index]*len(p_toks) + r_toks + [EOS]`.
   - Directly guarantees the causal invariant: $\text{input\_seq}[|P|]$ (last prompt token) targets $\text{target\_seq}[|P|] == r\_toks[0]$.

Both `SFTDataset.encode_example` and `DeterministicEvaluator.encode_record` now invoke these shared functions, eliminating potential implementation drift.

---

## 3. Deliberate Malformed Input Verification

We verified fail-closed safety behaviors under adverse inputs:
- **Empty response**: `DeterministicEvaluator.encode_record("Prompt\n", "")` returns `SKIPPED_EMPTY_RESPONSE` with empty input/target tensors.
- **Overlength sequence**: Returns `SKIPPED_OVERLENGTH`.
- **Invalid token values**: Rejects out-of-vocabulary token IDs.
- **Unreviewed dataset records**: Fails closed under `SFTDataset` trust gate if `review_status == "PENDING_HUMAN_REVIEW"` and `allow_exploratory_opt_in=False`.
