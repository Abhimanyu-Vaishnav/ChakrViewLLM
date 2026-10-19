# CHAKRVIEW SFT V3 READINESS REPORT
**Milestone:** SFT V3 Readiness, Data Quality & Native Capability Evaluation  
**Status:** Completed Engineering & Readiness Audit (Zero-Training Milestone)  
**Date:** 2026-10-19  
**Execution Environment:** Isolated `experiments/sft_v3_readiness/`

---

## 1. Executive Verdict

### **VERDICT: READY WITH BLOCKERS**

The ChakrView neural engineering stack has completed a comprehensive, rigorous diagnosis of the SFT V2 pilot, isolated the exact mathematical and structural root causes preventing native structured JSON generation, hardened the schema-constrained inference engine to 0% deadlocks, established a leakage-safe evaluation protocol, and defined an evidence-grounded SFT V3 curriculum specification.

However, **training SFT V3 must remain blocked until the dataset is completely regenerated according to the SFT V3 Curriculum Specification.** Training on the existing SFT V2 dataset or any similar ungrounded dataset is guaranteed to reproduce semantic hallucination and code-attractor collapse.

---

## 2. The Three Most Important Findings

### Finding 1: SFT V2 Suffered from Severe "Contextless Hallucination Forcing"
In SFT V2, training and validation prompts instructed the model to produce specific schema keys (e.g. `status`, `code`, `response_time_ms`) but provided **zero input context or factual source data** (e.g., *"Return a JSON object with fields 'status' and 'code' indicating server health status..."*). 
- The target values were arbitrary, ungrounded synthetic literals (e.g., `"queued"`, `202`, `14.5`).
- The neural network had no factual passage from which to extract answers. Consequently, the cross-entropy objective penalized the model unless it memorized arbitrary synthetic tuples.
- Under greedy decoding on validation prompts, the model had no basis to choose between `"standby"`, `"ready"`, or `"queued"`, yielding **0% semantic task accuracy**.

### Finding 2: Pretraining Code-Attractor Collapse Dominates Native Generation
ChakrMicro (3.4M parameters) was pretrained on code and language. When prompted to generate `{`, the model immediately falls into strong pretraining basins—specifically Python unit tests, doctests, and dictionary lookups (e.g., `{"__main__":\n import doctest`).
- In unconstrained generation, **85% of validation completions** immediately fell into Python boilerplate rather than structured entity extraction.
- In SFT V2, all training targets began with `{` followed immediately by newline and four spaces (`{\n    "status": ...`). This whitespace pattern matched Python dictionary definition syntax, actively triggering the pretraining code attractor.
- SFT V3 must adopt compact formatting, explicit anti-code conditioning, and grounded context passages.

### Finding 3: SFT V2 Pipeline & Token Alignment Were Mechanically Correct
The data leakage and tokenizer audit confirmed that the SFT V2 training pipeline was technically sound in its mechanics:
- **0 exact prompt overlaps**, **0 exact target overlaps**, and **0 normalized value leakages** between train and validation splits.
- Causal token sequence alignment, loss masking (`-100` for prompt tokens, valid target prediction for response tokens), and attention masks were verified to be **100% bug-free**.
- The maximum tokenized sequence length was 132 tokens (well within the 512-token context window), proving truncation was not responsible for output truncation.
- Therefore, the model's native failure was purely **semantic and data-distributional**, not an infrastructural bug.

---

## 3. Native Generation vs. Constrained Decoding Breakdown

A key mandate of ChakrView is never to misrepresent runtime decoding constraints as native model intelligence. We evaluated the candidate checkpoint (`sft_best_checkpoint.pt`) across three capability layers on the established 20-record validation set:

### Layer Comparison Matrix (20-Record Validation Split)

| Evaluation Metric | Layer A: Native Generation (Unconstrained) | Layer B: Schema-Trie Constrained (48 Tokens) | Layer B: Schema-Trie Constrained (96 Tokens) |
| :--- | :--- | :--- | :--- |
| **JSON Opening Rate (`{`)** | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) |
| **Completed Root Object (`}`)** | **10.0%** (2/20) | **65.0%** (13/20) | **75.0%** (15/20) |
| **Valid JSON Syntax (`json.loads`)**| **0.0%** (0/20) | **50.0%** (10/20) | **50.0%** (10/20) |
| **Decoder Deadlock Rate** | N/A (unconstrained) | **0.0%** (0/20) | **0.0%** (0/20) |
| **Budget Exhaustion Rate** | 90.0% (18/20) | 35.0% (7/20) | 25.0% (5/20) |
| **Code-Attractor Contamination** | **85.0%** (17/20) | **0.0%** (0/20 - masked) | **0.0%** (0/20 - masked) |
| **Exact Task Match (EM)** | **0.0%** (0/20) | **0.0%** (0/20 - ungrounded) | **0.0%** (0/20 - ungrounded) |

### Key Insights from Layer Analysis:
1. **Inference Scaffolding Elevates Syntax, Not Truth**: The Schema Prefix Trie elevates syntactically valid JSON from **0% to 50%** and completed objects from **10% to 75%** by masking illegal tokens. However, because the underlying task lacked context, exact task match remained at 0%.
2. **Elimination of Deadlocks**: Prior post-fix decoder audits exhibited trailing-comma deadlocks where the decoder entered an unrecoverable state after all required schema keys were emitted. We hardened the trie to reject commas once all keys are satisfied, achieving **0.0% deadlocks**.
3. **Escaped Sequences Fixed**: Character-level stream processing was hardened to prevent escaped characters (`\"`, `\\`, `\n`) from prematurely triggering value-type starter revalidation.

---

## 4. Tests Executed and Empirical Results

All diagnostic and regression test suites executed cleanly on CPU without errors:

1. `tests/test_json_constrained_decoder.py`: **4/4 passed**
   - Verified bracket stack balancing, string literal escaping, colon/comma syntax, and deterministic fake-model completions.
2. `tests/test_schema_prefix_trie.py`: **5/5 passed**
   - Verified prefix-matching across allowed schema keys, rejection of unknown keys, duplicate key suppression, and state isolation.
3. `tests/test_fake_model_schema_decoder.py`: **5/5 passed**
   - Verified end-to-end schema constrained decoding using deterministic fake models on simulated vocabularies.
4. `tests/test_schema_decoder_hardening.py` *(New suite)*: **6/6 passed**
   - Hardening regression tests for trailing-comma suppression, escape sequence feeding, empty/whitespace handling, and budget exhaustion behavior.
5. **Total Test Suite**: **20/20 unit and regression tests PASSED (100%)**.

---

## 5. Artifact and Repository Integrity Verification

In strict accordance with the Master Task safety gates, zero protected files were modified:

| Protected Artifact / File | Canonical Path | Verification Status | Verified SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Canonical Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | **MATCHED (Bit-identical)** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Canonical Tokenizer Vocab** | `chakrview/tokenizer/vocab_4096/vocab.json` | **MATCHED (Bit-identical)** | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical Tokenizer Merges** | `chakrview/tokenizer/vocab_4096/merges.json` | **MATCHED (Bit-identical)** | `ada4b1dc713296c0ee9bc177d6ba48b7a6344d216503b41d24f0c4bb6322ad1c` |
| **Runtime Session** | `chakrview/runtime/session.py` | **ZERO DIFF** | Clean git working tree |
| **Held-out Test Split** | `experiments/batch3111_3150/test.jsonl` | **NEVER ACCESSED** | Untouched / Zero reads |
| **Training / Learning Steps** | Active processes | **ZERO STEPS** | No optimizer or backward pass executed |

---

## 6. What Remains Unresolved

1. **Native Value Calibration for Integers and Booleans**:
   Even with schema logit masking, the 3.4M model exhibits high entropy when generating numeric digits without context, occasionally outputting long sequences of numbers unless bounded by token budget.
2. **Template Family Diversity vs Model Capacity**:
   With only 3.44M parameters, ChakrMicro's capacity must be carefully rationed. An SFT curriculum that attempts to cover too many disparate domains simultaneously may cause catastrophic forgetting of basic English syntax. Empirical scaling tests are required during dataset design.
3. **Multi-Key Value Completion Budget**:
   At 48 tokens, 35% of valid multi-field objects exhaust budget before closing. While 96 tokens improves completion to 75%, ChakrMicro's inference on low-power CPUs requires optimizing target schema brevity (1 to 3 fields max) to maintain real-time responsiveness.

---

## 7. Recommended Next Step

**Milestone: SFT V3 Dataset Builder Implementation & Preflight Verification**
1. Author the automated dataset generation script (`build_sft_v3_dataset.py`) implementing the 6 grounded curriculum categories defined in `SFT_V3_CURRICULUM_SPEC.md`.
2. Generate an isolated, leakage-verified dataset (e.g., 200 train / 40 val records) with guaranteed template-family disjointness.
3. Run the automated preflight verification suite to validate loss-masking, vocabulary coverage, and context-target consistency before requesting training authorization.
