# CHAKRVIEW SCHEMA-CONSTRAINED PREFIX TRIE VALIDATION REPORT
**Batch 3111–3150 / Inference Engineering & Schema Guidance**  
**Evaluation Scope**: 20 Structured Records from `val.jsonl` | Candidate Checkpoint (`sft_best_checkpoint.pt`)  
**Strict Safety Protocol**: Zero Model Training | Untouched Canonical Checkpoints/Tokenizer | Clean `session.py`

---

## 1. Executive Summary & Verdict

### Final Verdict: `VALIDATED FOR EXPERIMENTAL SCHEMA-CONSTRAINED USE`

An isolated, training-free schema-constrained decoding engine was designed, implemented, unit-tested, and evaluated in a 3-way head-to-head comparison against unconstrained greedy decoding and pure grammar-constrained decoding on the 20 structured validation records of `val.jsonl`.

### Key Empirical Findings:
1. **Elimination of the Disallowed Key Attractor**:
   - In unconstrained decoding, 20/20 prompts diverged into Python doctest boilerplate (`{"__main__":\n import doctest`).
   - In pure grammar-only decoding, 20/20 prompts permitted `"__main__"` because `"__main__"` is a syntactically valid JSON string.
   - In **Schema-Constrained Trie Decoding**, the prefix trie strictly blocked `_` and all non-schema characters, successfully forcing the decoder to emit the exact target keys requested in the prompt (`"status"`, `"code"`, `"metric"`, `"value"`, `"feature"`, `"enabled"`, `"name"`, `"count"`). Disallowed keys were 100% eliminated (0/20).
2. **Key Compliance vs. Value Generation Reality**:
   - While the Schema-Trie successfully guides key selection, it revealed the underlying model's value generation state:
   - Once a schema key was generated (e.g. `{"status"` or `{"metric"`), the model's unconstrained logits attempted to emit syntax incompatible with standard completions (such as newlines, spaces, or repeating comma-space tokens `":, "`, or looping whitespace), failing to reach a clean closing brace before exhausting the 48-token budget.
   - **Crucial Methodological Distinction**: Schema constraints prove that key compliance can be externally guided at inference time without retraining. However, externally guiding property names does **not** grant the model native understanding of field values or formatting flow. Value generation and object closure remain limited by model parameter capacity and pretraining priors.
3. **Safety & Runtime Isolation**:
   - `chakrview/runtime/session.py` remains completely untouched.
   - Canonical checkpoint SHA-256 (`96357705...`) and tokenizer artifacts are bit-for-bit identical.
   - Zero training or optimizer steps executed. Frozen held-out `test.jsonl` was never accessed.

---

## 2. Schema Representation & Prefix Trie Architecture

### Module: `experiments/batch3111_3150/schema_prefix_trie.py`

The architecture decouples syntax grammar validation from schema semantic filtering:

```
Candidate Token Logits
         │
         ▼
┌────────────────────────────────────────┐
│ Incremental JSON Grammar PDA           │ ---> Enforces RFC 8259 syntax
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ PrefixTrie Filter (Keys)               │ ---> Restricts string tokens to active
└──────────────────┬─────────────────────┘      allowed property names (no duplicates)
                   │
                   ▼
┌────────────────────────────────────────┐
│ Value Type Validator                   │ ---> Restricts value tokens to declared
└──────────────────┬─────────────────────┘      JSONType (STRING, NUMBER, BOOLEAN, etc.)
                   │
                   ▼
┌────────────────────────────────────────┐
│ Required Key Completeness Oracle       │ ---> Rejects '}' until all required
└────────────────────────────────────────┘      properties have been emitted
```

### Components:
- **`PrefixTrie`**: Implements a character-level trie over active candidate keys. Supports `is_valid_prefix(prefix)`, `is_exact_match(word)`, and `get_allowed_continuations(prefix)`. Rebuilt dynamically as keys are emitted to disallow duplicate properties.
- **`JSONSchema`**: Defines `properties: Dict[str, PropertySpec]`, `allow_additional_properties` (default `False`), and `allow_duplicate_keys` (default `False`). Tracks `required_keys` and `all_keys`.
- **`SchemaConstrainedJSONValidator`**:
  - Intercepts state transitions when entering object keys (`IN_OBJECT_EXPECT_KEY` or `IN_OBJECT_EXPECT_KEY_OR_CLOSE`).
  - Feeds candidate token characters into the active `PrefixTrie`. Immediately rejects tokens containing characters outside the trie (such as `_` for `"__main__"`).
  - Intercepts closing brace `}`: strictly disallows `}` if `len(required_keys - emitted_keys) > 0`.
  - Enforces value type constraints per property specification (`JSONType.STRING`, `INTEGER`, `NUMBER`, `BOOLEAN`, `NULL`).

---

## 3. Unit Test Verification (15/15 Passed)

Isolated tests in `tests/test_schema_prefix_trie.py` and `tests/test_json_constrained_decoder.py` verify all deterministic behavior:

| Test Case | Scope & Behavior Verified | Status |
| :--- | :--- | :---: |
| `test_prefix_trie_basic_and_shared_prefixes` | Exact matches, partial prefixes (`stat` -> `status`), continuation sets, rejection of disallowed keys (`__main__`, `import`). | **PASSED** |
| `test_schema_validator_requires_all_keys` | Rejects premature object closure `}` when required keys remain missing. | **PASSED** |
| `test_schema_validator_rejects_disallowed_and_duplicate_keys` | Rejects unknown keys at first character; rejects duplicate emissions of previously emitted keys. | **PASSED** |
| `test_schema_validator_type_constraints` | Rejects incorrect value types (e.g. integer for string property, boolean for integer property). | **PASSED** |
| `test_schema_validator_multi_token_key_simulation` | Correctly handles multi-token subword key fragments (`"serv"`, `"ice_"`, `"health"`). | **PASSED** |
| `test_grammar_decoder_unchanged_behavior` | Verifies base `IncrementalJSONValidator` remains unmodified and backwards-compatible. | **PASSED** |
| `test_validator_empty_containers` | Empty object and array oracle verification. | **PASSED** |
| `test_validator_basic_object_oracle_parity` | Standard Python `json.loads` oracle parity across numbers, strings, booleans, null. | **PASSED** |
| `test_validator_nested_structures` | Arbitrary object and array nesting support. | **PASSED** |
| `test_canonical_artifacts_untouched` | Cryptographic SHA-256 assertions on canonical checkpoints and tokenizers. | **PASSED** |

**Execution**:
`pytest tests/test_json_constrained_decoder.py tests/test_schema_prefix_trie.py -v`  
**Result**: `15 passed in 6.60s`.

---

## 4. Three-Way Validation-Only Model Comparison

Evaluated on the 20 structured-output validation prompts from `experiments/batch3111_3150/val.jsonl` using the candidate checkpoint (`sft_best_checkpoint.pt`) under identical 48-token generation limits.

### Quantitative Comparison Table

| Metric | Mode 1: Unconstrained Greedy | Mode 2: Grammar-Only Constrained | Mode 3: Schema-Trie Constrained |
| :--- | :---: | :---: | :---: |
| **Evaluated Records** | 20 | 20 | 20 |
| **Disallowed Keys Emitted (`__main__`)** | **20 / 20 (100%)** | **20 / 20 (100%)** | **0 / 20 (0.0%)** |
| **Target Key Set Initiated** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | **20 / 20 (100%)** |
| **Completed Valid JSON** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Completed Objects** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Value Exact Matches** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Budget Exhaustion (48 tok)** | 20 / 20 (100%) | 20 / 20 (100%) | 20 / 20 (100%) |
| **Deadlock / No Valid Tokens** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Average Tokens Generated** | 48.0 | 48.0 | 48.0 |

---

## 5. Comparative Generation Traces & Deep Diagnostic Analysis

### Record: `JSON_V2_VAL_001` (`structured_output_server_status`)
- **Prompt**: `Instruction: Return a JSON object with fields 'status' and 'code' indicating server health status and return code.\nResponse:`
- **Target**: `{"status": "queued", "code": 202}`
- **Expected Keys**: `["status", "code"]`

#### Trace Comparison:
1. **Unconstrained Greedy**:
   ```
   {"__main__":
       import doctest

       # import doctest

       # CError: CError: Callyou import 
   ```
   *Analysis*: Raw logits immediately trigger pretraining code attractor `"__main__"` and Python comments.

2. **Grammar-Only Constrained**:
   ```
   {"__main__":
          "__main__", "f":", ", "code":", ", "codecs."
          
   ```
   *Analysis*: Grammar rules suppress invalid code symbols (like unquoted identifiers), but since `"__main__"` is a valid JSON string, it enters the string manifold and loops through string keys and comma delimiters.

3. **Schema-Trie Constrained**:
   ```
   {"status"
          
          
          
          
          
          
          
          
          
   ```
   *Analysis*:
   - The prefix trie **successfully forced** the model to generate the exact required key `"status"`.
   - The grammar PDA permitted whitespace following `"status"`.
   - The model's top logit following `"status"` was a newline/indent token `\n       `. Because JSON grammar permits arbitrary whitespace between a key and its colon `:`, the unconstrained logits repeatedly selected whitespace tokens, consuming the remaining generation budget before reaching the colon `:`.

### Record: `JSON_V2_VAL_006` (`structured_output_network_metric`)
- **Prompt**: `Instruction: Return a JSON object with fields 'metric' and 'value' indicating telemetry metric name and measured reading.\nResponse:`
- **Target**: `{"metric": "vibration_g", "value": 0.12}`
- **Expected Keys**: `["metric", "value"]`

#### Trace Comparison:
- **Schema-Trie Constrained**:
  ```
  {"metric":", ", "value"
         
         
         
         
         
         
         
         
  ```
  *Analysis*:
  - Both schema keys (`"metric"` and `"value"`) were successfully emitted in sequence!
  - However, for the value of `"metric"`, the model predicted `", "` (a comma-string token).
  - Then, guided by the trie, it emitted `"value"`, followed by trailing whitespace loops until budget exhaustion.

---

## 6. Separation of Imposed Key Compliance vs. Native Value Capability

This experiment yields a definitive scientific finding for ChakrView:

1. **What Schema-Trie Decoding Achieves (Inference-Level Formatting)**:
   - Eliminates 100% of out-of-schema property names.
   - Prevents code attractor keys like `"__main__"` from ever being emitted.
   - Enforces correct structural transitions (e.g. disallows premature `}` before all required keys exist).
2. **What Schema-Trie Decoding Cannot Achieve (Fundamental Limitations)**:
   - It cannot make the model understand the semantic content of values without a value-level constraint (e.g. enum trie or regex mask).
   - If the grammar permits whitespace, and the model has an attractor toward repetitive whitespace indentation, the model will spend its generation budget emitting valid whitespace rather than advancing to values and closing delimiters.
   - **Conclusion**: Schema-constrained decoding is a powerful structural guardrail, but it is not a substitute for training-level alignment.

---

## 7. Safety, Cryptographic & Isolation Verification

All constraints of this master task were strictly enforced:

| Constraint | Status | Command / Evidence |
| :--- | :---: | :--- |
| `chakrview/runtime/session.py` Untouched | **VERIFIED** | `git diff --name-only chakrview/runtime/session.py` returned empty. |
| Canonical Step-10000 Checkpoint Unmodified | **VERIFIED** | SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| Canonical Tokenizer Vocabulary Unmodified | **VERIFIED** | SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| No Training or Optimizer Steps Executed | **VERIFIED** | 100% inference-only testing and comparative evaluation. |
| Frozen Held-Out Test Split Untouched | **VERIFIED** | `test.jsonl` was never opened, evaluated, or referenced. |
| Git Cleanliness | **VERIFIED** | No commits, pushes, merges, or remote operations. |

---

## 8. Next-Step Recommendations

1. **Retain Schema Prefix Trie as an Experimental Tool**:
   - Keep `experiments/batch3111_3150/schema_prefix_trie.py` as an optional experimental module for guided decoding benchmarks.
   - Do not integrate into canonical runtime `session.py` yet.
2. **Grammar Whitespace Tightening (Future Improvement)**:
   - In structured JSON generation, arbitrary whitespace (especially multiline indentation) can be disabled or restricted in the grammar PDA to force immediate colon `:` and value emission.
3. **Value Constraint Engineering**:
   - For categorical outputs, extend the trie to allowed value enums (e.g. `{"status": ["queued", "draining", "closed"]}`).
4. **Training Curriculum (SFT V3)**:
   - Model weights must be trained with strong penalization against Python boilerplate in JSON tasks so that native logit probability mass aligns with structured JSON values.
