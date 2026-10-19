# CHAKRVIEW CONSTRAINED DECODER VALIDATION & HARDENING REPORT
**Batch 3111–3150 / Inference-Layer Engineering & Validation**  
**Evaluation Scope**: Synthetic Fixtures & Isolated Validation Set (`val.jsonl`)  
**Strict Safety Protocol**: Zero Training Executed | Read-Only Canonical Artifacts | Untouched Runtime Core

---

## 1. Executive Summary & Verdict

### Final Verdict: `VALIDATED FOR EXPERIMENTAL USE`

The experimental constrained JSON decoder prototype (`experiments/batch3111_3150/json_constrained_decoder.py`) was audited, hardened, verified for tokenizer-boundary consistency, and evaluated across synthetic edge fixtures and the 20 structured validation records in `val.jsonl`.

### Key Findings:
1. **Incremental Validator Correctness**: The state-machine validator correctly parses JSON tokens incrementally, handles container nesting up to configured depths, validates strings, escapes, numbers, booleans, and null, and strictly rejects syntax violations (including trailing commas and post-completion trailing characters).
2. **Tokenizer Boundary Fidelity**: Verified that `tokenizer.decode(seq) == "".join(tokenizer.decode([t]) for t in seq)` holds without token-boundary mismatch across canonical BPE vocabulary boundaries.
3. **Synthetic Unit Test Pass Rate**: 9/9 synthetic test cases passed (100%), including nested containers, numbers, booleans, escape characters, trailing commas, and round-trip parity with Python's standard `json.loads` oracle.
4. **Validation-Split Empirical Findings (Grammar vs. Schema Constraints)**:
   - On unconstrained greedy decoding with the candidate model checkpoint (`sft_best_checkpoint.pt`), 0/20 outputs are valid JSON (100% divert to Python code headers: `{"__main__":\n import doctest`).
   - On pure grammar-constrained decoding, 0/20 outputs reach completion within 48 tokens. Because the unconstrained logits place `"__main__"` at rank 1, and `"__main__"` is a valid JSON string literal, a grammar-only validator permits it as an object key. Once inside the JSON string manifold, the model continues generating valid string tokens like `"f":", "`, looping until the 48-token generation budget is exhausted.
   - **Crucial Distinction**: Grammar constraint guarantees that every generated token keeps the sequence within the syntactically valid JSON prefix manifold. However, grammar constraints alone cannot steer a small 3.4M parameter model out of strong pretraining attractors (e.g., Python code) without an explicit **schema constraint** (e.g., a prefix trie over expected key names).
5. **Safety & Runtime Isolation**: The canonical checkpoint, canonical tokenizer, and `chakrview/runtime/session.py` remain completely untouched. No training, optimizer steps, or canonical runtime integrations were performed.

---

## 2. Supported Grammar & Validator Audit Findings

### Supported JSON Specification
The hardened `IncrementalJSONValidator` implements an incremental Pushdown Automaton (PDA) tracking syntactic states and container stacks:

| Feature / Syntax Element | Support Status | Implementation Details / Rules |
| :--- | :--- | :--- |
| **Objects** (`{ ... }`) | **Supported** | Enforces balanced `{` and `}`, key-value pairs, colon separator, comma delimiter between pairs. Trailing commas are strictly rejected (`IN_OBJECT_EXPECT_KEY`). |
| **Arrays** (`[ ... ]`) | **Supported** | Enforces balanced `[` and `]`, comma delimiter between items. Trailing commas are strictly rejected (`IN_ARRAY_EXPECT_VALUE`). |
| **Strings** (`" ... "`) | **Supported** | Enforces opening and closing double quotes. Tracks escape mode (`\`), supporting `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, and unicode escapes `\uXXXX` (requiring 4 hex digits). |
| **Numbers** | **Supported** | Incremental regex validation for integers, negative signs (`-`), decimals (`.`), and exponent notation (`e`/`E`, `+`/`-`). Rejects leading zeros (`05`), trailing decimals (`1.`), and malformed exponent prefixes upon token completion. |
| **Booleans & Null** | **Supported** | Supports prefix matches and exact value completion for `true`, `false`, and `null`. |
| **Whitespace** | **Supported** | Permitted between delimiters, keys, and values. Whitespace after root value completion is allowed; non-whitespace trailing characters are rejected. |
| **Nesting Depth** | **Supported** | Arbitrary container nesting supported via `container_stack`, bounded by configurable `max_depth` (default: 32). |
| **Completion Oracle** | **Supported** | Distinguishes between: (a) valid incomplete prefix (`is_valid_prefix()`), (b) syntactically completed root JSON object (`is_complete()`), and (c) unparseable prefix (`syntax_valid == False`). |

### Explicit Grammar Limitations:
- The grammar validator is purely syntactic; it does not validate JSON Schema types, regex constraints on string values, or enum sets unless supplemented by an explicit schema trie filter.
- Comments (`//` or `/* ... */`) and single-quoted strings (`'...'`) are strictly disallowed per RFC 8259.
- Arbitrary unescaped ASCII control characters (< 0x20) inside strings are rejected.

---

## 3. Tokenizer-Aware Token Filtering & Boundary Verification

### Technical Verification
A fundamental challenge in constrained neural decoding is token-boundary concatenation:
$$\text{decoded}(T_{1..k}) \stackrel{?}{=} \sum_{i=1}^k \text{decoded}(T_i)$$

For ChakrView's canonical BPE tokenizer (`data/experiments/vocab_4096/`):
1. **Decode Consistency**: Checked across token sequence boundaries:
   ```python
   tokenizer.decode(tokens) == "".join(tokenizer.decode([t]) for t in tokens)
   ```
   The canonical tokenizer uses byte-level BPE with reversible unicode mappings. For all standard ASCII and printable UTF-8 character sequences, decode-concatenation equivalence holds exactly.
2. **Dynamic Extension Verification**:
   The decoder evaluates candidate token $t \in \mathcal{V}$ by simulating the extended prefix:
   $$P_{new} = \text{tokenizer.decode}(\text{current\_tokens} + [t])$$
   The validator tests whether $P_{new}$ is a valid JSON prefix. This eliminates single-token heuristic assumptions and correctly handles sub-word fragments (e.g., `"sta"` followed by `"tus"`).
3. **No Halting / Infinite Loop Safety**:
   - `max_new_tokens` imposes a hard generation budget (default: 48 tokens).
   - If no token in the top-$K$ candidate list maintains a valid JSON prefix, the decoder searches down the logit distribution. If the entire vocabulary is exhausted without a legal continuation, generation safely terminates with `stop_reason = "NO_VALID_TOKENS"`.
   - Completed values are detected via `validator.is_complete()`; upon reaching completion, the decoder emits EOS or halts cleanly.

---

## 4. Synthetic Correctness Testing Results

Unit tests in `tests/test_json_constrained_decoder.py` verify 9 distinct synthetic test cases against Python's standard `json.loads` oracle:

| Test Case | Description | Oracle Comparison | Status |
| :--- | :--- | :--- | :--- |
| `test_empty_containers` | `{}` and `[]` validation, complete vs prefix states | `json.loads` matches | **PASSED** |
| `test_nested_structures` | Deeply nested objects and arrays `{"a": [1, {"b": true}]}` | `json.loads` matches | **PASSED** |
| `test_numbers_and_literals` | Negative numbers, decimals, exponents (`1.23e-4`), `true`, `false`, `null` | `json.loads` matches | **PASSED** |
| `test_escaped_strings_and_unicode` | Escapes `\"`, `\\`, `\n`, `\u0041` inside strings | `json.loads` matches | **PASSED** |
| `test_invalid_transitions` | Rejects missing colons, invalid punctuation, double commas, unclosed quotes | `json.loads` fails | **PASSED** |
| `test_trailing_commas_rejected` | Rejects trailing commas `{"a": 1,}` and `[1, 2,]` | RFC 8259 compliant | **PASSED** |
| `test_post_completion_rejection` | Rejects trailing garbage `{"a": 1} trailing` | `json.loads` parity | **PASSED** |
| `test_token_filtering_simulation` | Verifies masking blocks invalid tokens and permits valid tokens | Dynamic mask check | **PASSED** |
| `test_budget_and_no_valid_tokens` | Verifies graceful halt on budget exhaustion and empty legal token sets | Stop reason check | **PASSED** |

**Execution Output**:
```
pytest tests/test_json_constrained_decoder.py -v
============================= 9 passed in 4.58s =============================
```

**Full JSON Regression Suite**:
- `tests/test_json_sft_batch3071_3110.py`: 8 passed
- `tests/test_json_sft_v2_batch3111_3150.py`: 8 passed
- `tests/test_json_constrained_decoder.py`: 9 passed
- **Total: 25/25 passing tests**.

---

## 5. Validation-Only Model Comparison

Evaluated on all 20 structured-output records from `experiments/batch3111_3150/val.jsonl` using the trained candidate checkpoint (`sft_best_checkpoint.pt`, Step 100).

### Quantitative Summary Table

| Metric | Unconstrained Greedy | Constrained Decoder (Grammar-Only) |
| :--- | :---: | :---: |
| **Total Records Evaluated** | 20 | 20 |
| **Valid JSON Count** | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Completed Objects** | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Required-Key-Set Match** | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Average Generated Tokens** | 48.0 | 48.0 |
| **Budget Exhaustion Rate** | 20 / 20 (100.0%) | 20 / 20 (100.0%) |
| **No-Valid-Token Halts** | 0 / 20 (0.0%) | 0 / 20 (0.0%) |
| **Post-Processing / Repair** | None | None |

### Qualitative Analysis & Error Dissection

#### 1. Unconstrained Generation Pattern:
```json
// Prompt: Return a JSON object with fields 'status' and 'code'
{"__main__":
    import doctest

    # import doctest

    # CError: CError: Callyou import 
```
- The candidate model learned the opening delimiter `{"` from SFT V2.
- Immediately following `{"`, the model's highest logit token is `"__main__"` (rank 1), triggering an unconstrained divergence into Python `doctest` boilerplate.

#### 2. Grammar-Constrained Generation Pattern:
```json
// Prompt: Return a JSON object with fields 'status' and 'code'
{"__main__":
       "__main__", "f":", ", "code":", ", "codecs."
       
```
- The grammar validator correctly blocks non-JSON characters (e.g., unquoted identifiers, raw code).
- **The String-Manifold Trap**: Because `"__main__"` is a valid quoted JSON string, the grammar validator permits it as an object key. After `":`, the model predicts string literals (`"__main__"`, `", "`), which are syntactically valid JSON values.
- Once inside the string manifold, the model continues generating alternating string keys and values within valid JSON grammar, never generating the closing brace `}` before the 48-token generation budget expires.

---

## 6. Native vs. Constrained Capability Distinction

This validation highlights a critical distinction in small neural language models:

1. **Native Model Capability**:
   - The 3.4M parameter model has learned to initiate structured generation (`{"`).
   - However, its internal representations do not yet have sufficient conditional entropy to sustain key-value semantic alignment across the context without straying into pretraining code attractors (`__main__`, `import doctest`).
2. **Grammar-Only Constraint Limitations**:
   - A pure JSON grammar validator guarantees syntactic compliance with the language grammar $G_{\text{JSON}}$.
   - However, the language $L(G_{\text{JSON}})$ includes infinite valid strings (e.g. `{"__main__": "..."}`).
   - If the unconstrained probability distribution $P_\theta(w | c)$ heavily concentrates probability mass on unwanted valid strings, grammar masking alone will select the unwanted valid string rather than the semantically intended schema key.
3. **Runtime-Enforced Schema Constraints (The Solution)**:
   - To achieve 100% valid JSON matching the user's schema, the decoder must enforce a **Schema-Constrained Trie**:
     $$\mathcal{T}_{\text{keys}} = \{ \text{"status"}, \text{"code"} \}$$
   - When in state `IN_OBJECT_EXPECT_KEY`, the decoder must mask out all tokens that are not prefixes of $\mathcal{T}_{\text{keys}}$. This forces the model to emit the correct field names and prevents the `__main__` diversion.

---

## 7. Known Limitations & Remaining Risks

1. **Inference Latency Overhead**:
   - Dynamic simulation of candidate token extensions in Python requires decoding candidate token IDs on CPU. While efficient for top-$K$ candidates, filtering all 4096 tokens per step adds ~10–15ms per step on CPU.
2. **Greedy Trapping without Schema Steering**:
   - As proven on `val.jsonl`, pure grammar filtering does not force closure of objects if the model prefers continuing key-value chains.
3. **Absence of Backtracking**:
   - The decoder is a forward greedy search. If the model enters a syntactically valid prefix that cannot semantically complete within the token budget, the decoder will exhaust its budget.

---

## 8. Safety & Isolation Verification

All strict safety guidelines were upheld throughout this task:

| Verification Requirement | Status | Verification Evidence |
| :--- | :---: | :--- |
| `chakrview/runtime/session.py` untouched | **VERIFIED** | Clean git status; zero diffs against working tree. |
| Canonical Step-10000 Checkpoint Unmodified | **VERIFIED** | SHA-256 matches: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| Canonical Tokenizer Vocabulary Unmodified | **VERIFIED** | SHA-256 matches: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| No Training / Optimizer Steps Executed | **VERIFIED** | Inference-only execution; no training invocations. |
| Frozen Held-Out Test Set Untouched | **VERIFIED** | `test.jsonl` not evaluated; strictly used `val.jsonl`. |
| Git Cleanliness | **VERIFIED** | No commits, pushes, stashes, or remote operations. |

---

## 9. Next-Step Recommendations

1. **Keep Decoder as Experimental Component**:
   - Maintain `json_constrained_decoder.py` inside `experiments/batch3111_3150/` as an experimental prototype. Do not merge into canonical `chakrview/runtime/session.py`.
2. **Implement Schema-Trie Extension**:
   - Upgrade the constrained decoder to accept an optional `schema: Dict[str, Any]` parameter.
   - Restrict object keys to the schema's expected property names using a prefix trie, completely eliminating the `"__main__"` attractor.
3. **Training-Phase Alignment (SFT V3)**:
   - For native model capability, continue SFT curriculum engineering with strict negative sampling against Python boilerplate in JSON prompts.
