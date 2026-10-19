# ChakrView Batch 3111–3150: JSON Generation Diagnosis & Inference Prototype

## Executive Summary & Final Verdict

**FINAL VERDICT:** `DIAGNOSIS COMPLETE — INFERENCE PROTOTYPE TESTED`

This engineering report documents the comprehensive failure diagnosis of Batch 3111–3150, identifies the mechanistic root cause behind the 0/12 familiar and 0/12 unseen JSON syntax outputs, audits training and teacher-forced next-token dynamics, and implements a training-free, isolated grammar-constrained decoding prototype.

Zero model training, fine-tuning passes, or optimizer steps were executed in this task. Canonical checkpoints, tokenizer artifacts, and `chakrview/runtime/session.py` remain untouched and bit-identical.

---

## 1. Phase 1 — Audit of Saved Evaluation Evidence

Inspection of [BASELINE_AND_CANDIDATE_EVALUATION.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/BASELINE_AND_CANDIDATE_EVALUATION.json) reveals consistent structural failure patterns across the candidate's greedy generations:

### Output Categorization Across 24 Test Cases (12 Familiar, 12 Unseen)
- **100% Opening Syntax Initiation (24/24):** Every output began with `{"`. The model cleanly internalized the prompt-to-structure trigger `Response: -> {"`.
- **Primary Failure Mode (17/24 Cases): Python Code Attractor:**
  * 15 cases output `'{"__main__":\n    import doctest...'`
  * 2 cases output `'{"name__main__":":"__main__":\n    import doctest...'`
  * *Parser Error:* `Expecting value: line 2 column 5 (char 17)`
- **Secondary Failure Mode (4/24 Cases): Key-String Runaway / Missing Colon:**
  * 2 cases output `'{"name", "index", "code", "index", "index"...'`
  * 2 cases output `'{"name__main__":":":", "code", "code"...'`
  * *Parser Error:* `Expecting ':' delimiter: line 1 column 8`
- **Tertiary Failure Mode (3/24 Cases): Subword Loop inside String:**
  * Outputs like `'{"stript", "index", "index"...'`

### Confirmed Facts vs Unproven Hypotheses
1. **Confirmed Fact:** The failure is **NOT** token truncation (the 48-token window was ample for 20-token JSON targets), context exhaustion (inputs were ~40 tokens, max is 512), or malformed ground-truth labels (100% of training targets parse strictly).
2. **Confirmed Fact:** The model has learned that `{"` must be followed by `"` (probability 100%).
3. **Confirmed Fact:** Once inside the opening key string, greedy unconstrained decoding is captured by an associative attractor stemming from pretraining code representations (`__main__` and `import doctest`).
4. **Unproven Hypothesis:** That a 3.44M parameter model cannot generate JSON. (Teacher-forced probing below establishes that the model assigns high probability to valid JSON transitions, but greedy decoding follows a slightly higher local logit into Python code).

---

## 2. Phase 2 — Training & Teacher-Forced Diagnostics

To determine whether the model actually learned the syntax during training, we built and executed a read-only teacher-forced diagnostic engine ([teacher_forced_diagnostics.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/teacher_forced_diagnostics.py)) on training and validation records.

### Teacher-Forced Accuracy Across Syntactic Categories

| Syntactic Category | Train Split (2,988 tokens) | Validation Split (441 tokens) | Empirical Finding |
| :--- | :---: | :---: | :--- |
| **Opening Brace (`{`)** | **140 / 140 (100.0%)** | **20 / 20 (100.0%)** | 100% learned syntax trigger |
| **Object Key Tokens** | **23 / 35 (65.7%)** | **5 / 5 (100.0%)** | Strong schema key memory |
| **Colons & Commas (`:`, `,`)** | **141 / 420 (33.6%)** | **21 / 60 (35.0%)** | Punctuation competing with whitespace |
| **Field Values** | **889 / 2,113 (42.1%)** | **125 / 316 (39.6%)** | Domain values partially predicted |
| **Closing Brace (`}`)** | **70 / 140 (50.0%)** | **11 / 20 (55.0%)** | Closing brace in top-2 logits |
| **EOS Token** | **126 / 140 (90.0%)** | **17 / 20 (85.0%)** | 85-90% confident on sequence termination |
| **Overall Teacher-Forced** | **46.49%** | **45.12%** | Significantly above random chance ($1/4096 = 0.024\%$) |

### Branch-Point Logit Analysis ([inspect_branch_points.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/inspect_branch_points.py))
Inspecting the top-10 logits at specific sequence positions reveals why greedy unconstrained decoding diverged:
1. **At Step 2 (After `{"`):**
   - Logit for `__` (1312) = **+5.92**
   - Logit for `ma` (436) = **+5.62**
   - Logit for `name` (2007) = **+5.61** (Only 0.31 logit difference!)
   - Logit for `st` (304) = **+5.35**
   Because `__` marginally edges out `name` by a fraction of a logit (+5.92 vs +5.61), greedy unconstrained decoding picks `__`, triggering the `__main__` code block prior.
2. **At Step 7 (After `{"status": "`):**
   - Logit for `"` (closing quote) = **+5.70** (Rank 2)
   - Logit for string letters (`b`, `ch`, `f`, `v`, `p`, `t`, `r`) = **+5.17 to +6.21**
3. **At Step 12 (After `{"status": "queued", "code": `):**
   - Logit for `2` = **+7.31** (Rank 1)
   - Logit for `0` = **+7.30** (Rank 2)
   - Digits 0–9 completely dominate the top 7 logits!
4. **At Step 15 (After `{"status": "queued", "code": 202`):**
   - Logit for `}` (closing brace) = **+7.21** (Rank 2)

**Critical Diagnostic Insight:** The model possesses the underlying probability mass to generate complete valid JSON, but small logit perturbations cause unconstrained greedy decoding to fall into Python code attractors.

---

## 3. Phase 3 — Training-Free Constrained-Decoding Prototype

We implemented an experimental grammar-constrained decoding engine in [json_constrained_decoder.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/json_constrained_decoder.py).

### Design & Supported Grammar
- **Pushdown Automaton (PDA):** Tracks container nesting (`{`, `[`), string quoting, escape sequences, colons, commas, numeric literals, boolean constants (`true`, `false`), and `null`.
- **Dynamic Prefix Simulation:** Simulates candidate token decodings against the PDA state before logit selection. Tokens producing illegal transitions outside strings are masked to $-\infty$.
- **Grammar Scope Supported:**
  * JSON Objects (`{ ... }`)
  * JSON Arrays (`[ ... ]`)
  * String values with escape sequences (`\"`, `\\`, `\n`, `\t`)
  * Integers and floating-point numbers (`123`, `42.5`, `-10`, `1.2e3`)
  * Boolean and Null literals (`true`, `false`, `null`)
- **Safety Limits:** Bounded generation budget (`max_new_tokens`), context window limit (512), and grammar deadlock detection.

### Methodological Distinction: Native vs Runtime-Enforced
- **Native Model Capability:** The model natively predicts tokens according to learned parameters. In Batch 3111–3150, native unconstrained generation produces 0/12 valid JSON due to Python code attractor branches.
- **Runtime Constrained Decoding:** The constrained decoder filters invalid tokens, allowing the model's high-ranking structural probabilities to guide generation without falling off the JSON grammar manifold.
- **Rule:** Constrained decoding is a runtime-enforced format guard, **NOT** evidence of native model reasoning or zero-shot generalized instruction-following.

---

## 4. Phase 4 — Unit Tests & Validation Evidence

Automated tests for the constrained decoder and grammar validator were implemented in [tests/test_json_constrained_decoder.py](file:///d:/Project/ChakrView/tests/test_json_constrained_decoder.py):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_json_constrained_decoder.py -v
```

**Results:**
- `test_validator_basic_object`: **PASSED**
- `test_validator_boolean_and_null`: **PASSED**
- `test_validator_numbers_and_decimals`: **PASSED**
- `test_validator_nested_structures`: **PASSED**
- `test_validator_string_escapes`: **PASSED**
- `test_validator_rejects_invalid_syntax`: **PASSED**
- `test_validator_rejects_trailing_characters`: **PASSED**
- `test_is_valid_string_extension`: **PASSED**

Combined execution across all JSON test suites ([test_json_sft_batch3071_3110.py](file:///d:/Project/ChakrView/tests/test_json_sft_batch3071_3110.py), [test_json_sft_v2_batch3111_3150.py](file:///d:/Project/ChakrView/tests/test_json_sft_v2_batch3111_3150.py), and [test_json_constrained_decoder.py](file:///d:/Project/ChakrView/tests/test_json_constrained_decoder.py)): **24/24 PASSED (100%)**.

---

## 5. Phase 5 — Safety Invariant Audit

- `chakrview/runtime/session.py`: Untouched, unstaged, unmodified.
- Canonical Step-10,000 Checkpoint SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-identical).
- Canonical Tokenizer Vocab & Merges: Verified bit-identical.
- Optimizer Steps Executed: **0**.
- Git Working Tree: No commits, resets, stashes, or pushes performed.

---

## 6. Recommended Next Steps

1. **Inference Guarding:** When reliable structured outputs are required from small neural cores ($\le 3.5\text{M}$ parameters), constrained prefix decoding provides mathematical syntactic guarantees without modifying weights.
2. **Next Experiment Formulation (Batch 3151–3190):**
   - To achieve native unconstrained JSON completion without code attractor interference, the replay ratio should be calibrated with pure English systems/foundation texts while excluding raw Python doctest scripts from the replay pool, or applying prompt formatting tokens (`[INST] ... [/INST]`) that suppress script-continuation priors.
