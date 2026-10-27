# EXPERIMENT 31 PROTOCOL
## Execution Steps, Data Integrity Guarantees & Audit Trails

### 1. Data Generation & Disjointness Enforcement
1. **Vocabulary Space:**
   - Base Tokenizer: `chakrview/brain/tokenizers/chakrmicro_bpe_v1/`
   - Training Values: English uppercase letters `['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']` (12 tokens).
   - Unseen Held-out Values: `['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!']` (24 tokens).
   - Abstention Symbol: `'?'`.
2. **Disjointness Audit:**
   - Programmatically assert `set(train_values).isdisjoint(set(unseen_values))`.
   - Programmatically assert each token encodes to a single BPE token ID in `ChakrTokenizer`.
   - Ensure negative examples do not accidentally contain the queried key.

### 2. Experimental Model Implementation Rules
- Canonical model `ChakrMicro` imported directly from `chakrview.brain.model` in read-only mode.
- Experimental variant `ChakrPointerMicro` extends `ChakrMicro` by tapping the final hidden state to compute copy logits over the context sequence, using an interpolation gate $p_{gen} = \sigma(W_g h + b_g)$ combining vocabulary logits and copy distribution.
- All experimental classes live strictly under `experiments/exp31_capability_generalization/`.

### 3. Training & Evaluation Pipeline
1. Run dataset generator: `generate_data.py`.
2. Validate dataset disjointness and save SHA-256 manifests.
3. Evaluate Variant 0 (Canonical Baseline Zero-Shot) across all 6 test groups.
4. Train Variant 1 (Canonical SFT) on CPU for 300 steps.
5. Evaluate Variant 1 on all 6 test groups.
6. Train Variant 2 (ChakrPointerMicro) on CPU for 300 steps.
7. Evaluate Variant 2 on all 6 test groups.
8. Run full regression suite (`tests/test_v060_release_smoke.py`).
9. Verify integrity hashes of protected and canonical files.
10. Generate full audit report and artifact manifest.
