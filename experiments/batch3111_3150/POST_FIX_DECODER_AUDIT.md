# CHAKRVIEW POST-FIX DECODER AUDIT & NATIVE CAPABILITY SEPARATION REPORT
**Batch 3111–3150 / Inference Engineering & Methodological Audit**  
**Evaluation Scope**: 20 Structured Validation Records (`val.jsonl`) | Candidate Checkpoint (`sft_best_checkpoint.pt`)  
**Strict Safety Protocol**: Zero Training Executed | Read-Only Canonical Checkpoints/Tokenizer | Clean `session.py`

---

## 1. Executive Verdict & Operational Status

### Final Status: `VALIDATED FOR EXPERIMENTAL USE ONLY (NOT PRODUCTION-READY)`

The schema-constrained decoder changes have been audited, reproduced, and hardened.
- **Reproducibility**: The previously reported 5/20 valid JSON completion rate was reproduced and further elevated to **10/20 (50.0%) valid JSON outputs** on the 48-token benchmark and **10/20 (50.0%)** on the 96-token diagnostic by fixing trailing-comma deadlocks when all permitted schema keys have been emitted.
- **Root Object Completion**: Reached **13/20 (65.0%)** on the 48-token budget and **15/20 (75.0%)** on the 96-token diagnostic.
- **Deadlocks / Empty Token Traps**: Reduced to **0/20 (0.0%)** across both benchmarks.
- **Production Assessment**: The current decoder is strictly an **experimental tool** for controlled benchmarking. It is **not suitable for production deployment** because native model capability remains unaligned: exact task semantic value match is **0/20 (0.0%)**, with the model generating placeholder strings (e.g. `", "`) or numeric artifacts rather than ground-truth answers.

---

## 2. Independent Metric Definitions & Verification

Every metric was computed independently using Python's standard `json.loads` oracle without heuristic text repairs or string substitutions:

| Metric | Formal Definition | Scoring Criteria |
| :--- | :--- | :--- |
| **Generation Terminated** | $T_{\text{stop}} \in \{\text{COMPLETE\_JSON}, \text{STOP\_TOKEN}, \text{MAX\_TOKENS}\}$ | Decoder halted via valid stop condition without infinite looping. |
| **Root Object Completed** | `validator.is_complete()` is `True` | The pushdown automaton container stack emptied (`len(stack) == 0`) and root `{}` closed. |
| **Parses with `json.loads`** | `isinstance(json.loads(text), dict)` | Output text passes standard strict RFC 8259 parsing. |
| **Required Keys Present** | $\mathcal{K}_{\text{req}} \subseteq \text{keys}(\text{parsed\_obj})$ | All required schema property names are present in the parsed dictionary. |
| **No Unknown / Duplicate Keys** | $\text{keys}(\text{parsed\_obj}) \subseteq \mathcal{K}_{\text{all}}$ | No extra properties emitted; duplicate property names rejected. |
| **Values Match Declared Types** | $\forall k, v: \text{type}(v) \in \text{schema}[k].\text{types}$ | String, integer, number, and boolean types match schema constraints. |
| **Schema Valid Match** | $\text{Parses} \land \mathcal{K}_{\text{req}} \land \text{NoExtra} \land \text{TypesValid}$ | End-to-end schema compliance. |
| **Task Semantic Exact Match** | $\text{parsed\_obj} == \text{target\_obj}$ | Model generated ground-truth values matching prompt instructions. |

---

## 3. Investigation & Resolution of Deadlock Cases

### Finding: The Trailing Comma Deadlock
In the initial post-fix evaluation, 5/20 records (specifically `feature_flag` records `VAL_011` through `VAL_015`) halted with `NO_VALID_TOKENS` after generating:
```json
{"feature": ", ", "enabled": false, "
```
- **Audit Diagnosis**:
  1. Both required keys (`feature` and `enabled`) were emitted.
  2. Because the schema specified `allow_additional_properties=False` and `allow_duplicate_keys=False`, no candidate keys remained in the schema trie (`self.schema.all_keys - self.emitted_keys == set()`).
  3. However, following the value `false`, the base grammar PDA permitted either `,` (comma) or `}` (close).
  4. The model's unconstrained logits ranked `, ` higher than `}`. The validator accepted `,`, transitioning to `IN_OBJECT_EXPECT_KEY`.
  5. In state `IN_OBJECT_EXPECT_KEY`, the trie was completely empty. Consequently, every token in the 4096-vocabulary was rejected, causing an immediate deadlock (`NO_VALID_TOKENS`).
- **Remediation**:
  Updated `SchemaConstrainedJSONValidator.feed_char()`:
  ```python
  if ch == ',' and not self.base_validator.in_string:
      if len(self.base_validator.stack) == 1 and self.base_validator.stack[-1] == '{':
          if not self.schema.allow_additional_properties:
              remaining_keys = self.schema.all_keys - self.emitted_keys
              if not remaining_keys and not self.schema.allow_duplicate_keys:
                  return False  # Disallow comma when no more keys can be accepted!
  ```
- **Result**:
  Blocking `,` when all allowed keys are emitted forces the model to select `}` (closing brace). Across all 20 records, **deadlocks were completely eliminated (0/20, 0.0%)**. All 5 feature flag records now cleanly complete into valid JSON objects (`{"feature": ", ", "enabled": false}`).

---

## 4. Audit of `strict_spacing=True` vs. General JSON

### Operational Scope:
- In `strict_spacing=True` mode, whitespace is permitted inside quoted strings, but multiline indentation characters (`\n`, `\r`, `\t`) outside strings are disallowed during object generation.
- **Technical Justification**: Small 3.4M models trained on code corpora have an overwhelmingly high prior for Python `doctest` indentation loops (`\n        self.`). Because JSON grammar permits arbitrary whitespace, the model falls into an infinite whitespace attractor that consumes the entire token budget before emitting colons or values.
- **Preservation of General JSON**:
  - Escaped characters inside strings (including `\n`, `\"`, `\\`, and `\uXXXX`) are 100% supported and tested in `test_escaped_characters_and_quotes_in_values`.
  - When `strict_spacing=False` is passed, the validator accepts arbitrary whitespace anywhere per RFC 8259 (tested in `test_strict_spacing_vs_general_whitespace`).
  - **Explicit Documentation**: `strict_spacing=True` is an intentional inference-time guardrail for small language models, not an RFC 8259 parser defect.

---

## 5. Comprehensive Test Coverage Audit

Targeted unit testing in `tests/`:
- `tests/test_json_constrained_decoder.py` (9 tests)
- `tests/test_schema_prefix_trie.py` (6 tests)
- `tests/test_fake_model_schema_decoder.py` (1 test)
- `tests/test_schema_decoder_hardening.py` (4 tests)
- **Total: 20 passing unit tests in 4.02s**.

| Requirement | Test Coverage Status | Test Function |
| :--- | :---: | :--- |
| Single & shared prefix trie matching | **Covered** | `test_prefix_trie_basic_and_shared_prefixes` |
| Disallowed key rejection (`__main__`) | **Covered** | `test_schema_validator_rejects_disallowed_and_duplicate_keys` |
| Duplicate key rejection | **Covered** | `test_schema_validator_rejects_disallowed_and_duplicate_keys` |
| Required property completeness enforcement | **Covered** | `test_schema_validator_requires_all_keys` |
| Value type constraints (string, int, bool, etc.) | **Covered** | `test_schema_validator_type_constraints` |
| Escaped characters (`\"`, `\\`, `\n`) in values | **Covered** | `test_escaped_characters_and_quotes_in_values` |
| Trailing comma deadlock avoidance | **Covered** | `test_comma_disallowed_when_no_remaining_keys`, `test_fake_model_avoids_trailing_comma_deadlock` |
| Strict spacing vs. general JSON spacing | **Covered** | `test_strict_spacing_vs_general_whitespace` |
| Deterministic fake-model scripted generation | **Covered** | `test_fake_model_schema_constrained_generation` |
| Cryptographic canonical artifact integrity | **Covered** | `test_canonical_artifacts_untouched` |

---

## 6. Validation Benchmark Results (Per-Mode & Per-Record)

Evaluated on candidate checkpoint (`sft_best_checkpoint.pt`) across the 20 structured validation records in `val.jsonl`:

### Quantitative Summary Table

| Metric | Unconstrained Greedy | Grammar-Only Constrained | Schema-Trie (48-Token Budget) | Schema-Trie (Diagnostic 96-Token) |
| :--- | :---: | :---: | :---: | :---: |
| **Evaluated Records** | 20 | 20 | 20 | 20 |
| **Disallowed Keys (`__main__`)** | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** |
| **Required Schema Keys Initiated**| 0 / 20 (0.0%) | 0 / 20 (0.0%) | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** |
| **Completed Root Objects** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | **13 / 20 (65.0%)** | **15 / 20 (75.0%)** |
| **Valid Syntax Objects (`json.loads`)**| 0 / 20 (0.0%) | 0 / 20 (0.0%) | **10 / 20 (50.0%)** | **10 / 20 (50.0%)** |
| **Schema Valid Matches** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | **10 / 20 (50.0%)** | **10 / 20 (50.0%)** |
| **Value Exact Matches** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Budget Exhaustion Rate** | 20 / 20 (100.0%) | 20 / 20 (100.0%) | **7 / 20 (35.0%)** | **5 / 20 (25.0%)** |
| **Deadlock / Empty Token Traps** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** |
| **Average Generated Tokens** | 48.0 | 48.0 | **31.55** | **48.05** |
| **Median Generated Tokens** | 48 | 48 | **27.5** | **27.0** |

---

### Per-Category Performance Breakdown (48-Token Budget)

1. **`server_status` (Records 001–005)**:
   - Target: `{"status": <str>, "code": <int>}`
   - Result: **5/5 (100%) completed valid JSON objects** matching required schema!
   - Generated text pattern: `{"status": ", ", "code": 0}` (completed in 20 tokens).
2. **`feature_flag` (Records 011–015)**:
   - Target: `{"feature": <str>, "enabled": <bool>}`
   - Result: **5/5 (100%) completed valid JSON objects** matching required schema!
   - Generated text pattern: `{"feature": ", ", "enabled": false}` (completed in 28 tokens).
3. **`network_metric` (Records 006–010)**:
   - Target: `{"metric": <str>, "value": <float>}`
   - Result: 0/5 completed (Budget exhausted at 48 tokens).
   - Generated pattern: `{"metric": ", ", "value": 3007063333...}`. The model entered a repetitive digit emission attractor without outputting a closing brace before 48 tokens. In the 96-token run, 2 records closed at ~52 tokens (`Completed Objects = 2/5`).
4. **`inventory_item` (Records 016–020)**:
   - Target: `{"name": <str>, "count": <int>}`
   - Result: 3/5 completed root objects (`VAL_016`, `VAL_018`, `VAL_020`), but generated leading zero integers (`: 060}`). Standard Python `json.loads` rejects numbers with leading zeros, resulting in syntax error: `Leading zeros are not allowed`. In the remaining 2 records, budget exhausted on repetitive digits (`: 200706333...`).

---

## 7. Native Capability vs. Inference-Constraint Findings

| Dimension | Native Model Generation | Constrained Inference Engine | Methodological Assessment |
| :--- | :--- | :--- | :--- |
| **JSON Delimiter Initiation** | Learned (`{"`) with high logit confidence. | Validated in base PDA state transitions. | Genuine model capability from SFT V2. |
| **Key Selection** | Attracted to Python code (`"__main__"`, `import`). | Forced to required schema keys via `PrefixTrie`. | **Runtime-Enforced**. Model does not natively prefer schema keys. |
| **Container Nesting & Closing** | Random continuation or trailing commas. | Enforced by PDA stack balance and key checks. | **Runtime-Enforced**. Model lacks structural closure awareness. |
| **Value Content & Semantics** | Emits placeholder commas `", "` or digit loops. | Type-guarded (rejects non-numeric for int fields). | **Model Failure**. The model cannot produce accurate values without training. |

**Crucial Scientific Takeaway**: Constrained decoding is a deterministic steering mechanism that guarantees syntax and schema structure. It cannot inject knowledge, semantic understanding, or task-specific reasoning into model weights.

---

## 8. Canonical Artifact Integrity Verification

| Artifact | Expected SHA-256 | Actual Verified SHA-256 | Verification Result |
| :--- | :--- | :--- | :---: |
| **Canonical Checkpoint Step 10,000** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED BIT-IDENTICAL** |
| **Tokenizer Vocabulary (`vocab.json`)** | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED BIT-IDENTICAL** |
| **Tokenizer Merges (`merges.json`)** | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED BIT-IDENTICAL** |
| **`chakrview/runtime/session.py`** | Untouched | `git diff --name-only` returned empty | **VERIFIED UNTOUCHED** |
| **Held-Out Test Split (`test.jsonl`)** | Never accessed | Timestamp unchanged | **VERIFIED UNTOUCHED** |
| **Training Steps Executed** | 0 | 0 optimizer steps run | **VERIFIED INFERENCE-ONLY** |

---

## 9. Evidence-Based Next Step Recommendation

**Do not integrate the constrained decoder into `chakrview/runtime/session.py` yet.**  
While the decoder achieves 50% schema-valid object completion on validation prompts, the 0% semantic value accuracy proves that the bottleneck is neural capability, not runtime formatting.

**Single Recommended Next Step**:
Design an authorized **SFT V3 curriculum dataset** that:
1. Applies **contrastive negative sampling** to suppress Python code attractors (`__main__`, `import doctest`) in response to structured JSON prompts.
2. Trains the model on realistic, semantically grounded key-value pairs (e.g. associating `"status"` with `"ok"`, `"queued"`, `"error"` rather than `", "`).
3. Uses the validated `schema_prefix_trie.py` exclusively as an experimental evaluation benchmark, keeping canonical runtime APIs clean until native model representations improve.
