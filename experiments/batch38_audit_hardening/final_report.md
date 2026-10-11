# CHAKRVIEW — BATCH 38 MASTER SPRINT FINAL REPORT
## Audit Tool Hardening, Executable Validation & Enforced SFT Readiness Gate

### Executive Summary

Batch 38 replaced keyword/lexical heuristics with a hardened 10-dimensional record audit, constructed an allowlisted safe behavioral execution harness for code examples, built a runtime-dynamic baseline evaluation harness, verified causal loss alignment and target shifting through synthetic tensor tests, implemented a fail-closed multi-dimensional SFT readiness gate, and performed a claim-by-claim audit of Batch 37 findings.

Zero Git mutations (`commit`, `push`, `rebase`, `reset`, `clean`) were performed. All historical checkpoints, datasets, and reports were preserved. No SFT training runs were launched.

---

### 1. Initial and Final Git State

- **Branch:** `dev/v0.6-v0.7-validation`
- **HEAD Commit:** `7c3be923acf997d98f693b3509bece86ddf97641`
- **Tracked Modifications Preserved:** `chakrview/runtime/session.py`
- **New Experiment Artifacts:** `experiments/batch38_audit_hardening/`
- **New Tests:** `tests/test_batch38_audit_hardening.py`
- **Git Mutation Operations:** 0 commits, 0 pushes, 0 merges, 0 rebases, 0 resets.

---

### 2. Canonical ChakrMicro Baseline Invariance

- **Parameter Count:** 3,443,136
- **Architecture:** 6 layers, d_model 192, 6 heads, hidden_dim 512, context 512, vocab 4096
- **Fresh-Initialization SHA-256:** `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Verification Status:** **CONFIRMED 100% INVARIANT**

---

### 3. Hardened Dataset Record-by-Record Findings

The audit evaluated all 19 Batch 36 expanded records across 10 strictly independent dimensions rather than collapsing them into an ambiguous "SUPPORTED" tag:

| Record ID | Split | Domain | Schema | File Exists | File SHA | Evidence Location | Exact Quote | Claim Support | Tech Correct | Code Exec | Dups | Human Review |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `EXP-TR-EN-01` | train | english_systems | PASS | PASS | PASS | PASS (L627) | FAIL | PARTIAL | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-EN-02` | train | english_systems | PASS | PASS | PASS | PASS (L663) | FAIL | **FAIL** | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-EN-03` | train | english_foundation | PASS | PASS | PASS | **FAIL** | FAIL | **FAIL** | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-HI-01` | train | hindi_foundation | PASS | PASS | PASS | PASS (L4-5) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-HI-02` | train | hindi_literature | PASS | PASS | PASS | PASS (L207) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-HI-03` | train | hindi_literature | PASS | PASS | PASS | PASS (L207) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-HG-01` | train | hinglish_foundation | PASS | PASS | PASS | PASS (L3,15,27) | FAIL | PARTIAL | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-HG-02` | train | hinglish_dialogue | PASS | PASS | PASS | PASS (L15) | FAIL | PARTIAL | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-CD-01` | train | code_foundation | PASS | PASS | PASS | PASS (L1-7) | PASS | PASS | PASS | **PASS** | PASS | NOT_VERIFIED |
| `EXP-TR-CD-02` | train | code_algorithms | PASS | PASS | PASS | PASS (L2-14) | FAIL | PASS | PASS | **PASS** | PASS | NOT_VERIFIED |
| `EXP-TR-MA-01` | train | math_foundation | PASS | PASS | PASS | PASS (L13) | PASS | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-TR-MT-01` | train | multiturn_dialogue | PASS | PASS | PASS | **FAIL** | FAIL | **FAIL** | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-EN-01` | val | english_systems | PASS | PASS | PASS | PASS (L4) | FAIL | PARTIAL | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-EN-02` | val | english_systems | PASS | PASS | PASS | PASS (L1) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-HI-01` | val | hindi_computing | PASS | PASS | PASS | PASS (L3) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-HI-02` | val | hindi_science_culture | PASS | PASS | PASS | PASS (L3) | FAIL | PASS | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-HG-01` | val | hinglish_tech | PASS | PASS | PASS | PASS (L5) | FAIL | PARTIAL | PASS | N/A | PASS | NOT_VERIFIED |
| `EXP-VL-CD-01` | val | code_python | PASS | PASS | PASS | PASS (L6) | PASS | PASS | PASS | **PASS** | PASS | NOT_VERIFIED |
| `EXP-VL-MA-01` | val | code_math | PASS | PASS | PASS | PASS (L3) | PASS | PASS | PASS | N/A | PASS | NOT_VERIFIED |

#### Aggregate Audit Summary
- **Schema Validity:** 19/19 PASS (100%)
- **Source File Existence:** 19/19 PASS (100%)
- **Source File SHA-256 Validity:** 19/19 PASS (100%)
- **Evidence Location Validity:** 17/19 PASS, 2 FAIL (`EXP-TR-EN-03`, `EXP-TR-MT-01`)
- **Exact Quotation Matching:** 4/19 PASS, 15/19 FAIL (Most responses are authored syntheses, not verbatim corpus quotes)
- **Factual Claim Support:** 11 PASS, 5 PARTIAL, 3 FAIL (`EXP-TR-EN-02` cited UDP telemetry for autoregressive decoding bandwidth; `EXP-TR-EN-03` cited text with no feed-forward theory; `EXP-TR-MT-01` cited text with no 512 context limit reference)
- **Technical Correctness:** 19/19 PASS (Textbook ML theory is correct even when ungrounded in raw text)
- **Executable Code Correctness:** 3 PASS, 0 FAIL, 16 N/A (Validated with actual function invocation on controlled test cases)
- **Duplicate & Near-Duplicate Risk:** 19/19 PASS (No identical prompt or response duplicates)
- **Human Review Status:** 0 REVIEWED, 19 NOT_VERIFIED (Explicitly held distinct from automated audits)

---

### 4. Safe Executable Code Validation Harness

In Batch 37, code validation was reported as `EXECUTION_PASS` by calling `exec()` on function definitions without invoking the functions.

Batch 38 built [`SafeCodeExecutionHarness`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/audits/validator.py):
1. **Behavioral Invocation Requirement:** Demonstrated in tests that a broken function (`return 1/0`) successfully compiles and defines via `exec()`, but immediately crashes upon invocation.
2. **Controlled Test Execution:**
   - `compute_cross_entropy` was executed on controlled logits (`[2.0, 1.0, 0.1]`) with targets 0 and 1, validating numerical outputs against $- \log(\text{softmax})$ ground truth within $10^{-5}$ precision.
   - `binary_search` was executed across 9 edge cases (mid target, first element, last element, missing elements, empty list, single element).
   - `assert d_model % n_heads == 0` was tested on both valid `(192, 6)` and invalid `(192, 5)` inputs to confirm that it raises `AssertionError`.
   - Untested or non-allowlisted code snippets return `NOT_VERIFIED`, never `EXECUTION_PASS`.

---

### 5. Runtime-Dynamic Baseline Evaluation Harness

Built [`HardenedBaselineEvaluator`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/evaluation/evaluator.py):
- **Dynamic File SHA-256:** Calculated from file bytes at runtime: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Size: 42,212,787 bytes).
- **Dynamic EOS Token Resolution:** Resolved `<EOS>` token ID directly from `data/experiments/vocab_4096/config.json` (`resolved_eos_id=1`), asserting exact match with expected ID.
- **Model Eval Mode Enforced:** `model.eval()` explicitly asserted prior to inference.
- **Context Limit Guard:** Enforced context-length boundary before forward pass.
- **Deterministic Greedy Decoding Results (10 Diagnostic Smoke Prompts):**
  - Total prompts: 10
  - First-token EOS count: **0 / 10 (0.0% rate)**
  - Empty decoded outputs: **0 / 10 (0.0% rate)**
  - Average generated tokens: **30.7**
  - Stop reason distribution: 9 `max_tokens`, 1 `eos`
  - Per-prompt token IDs, generated IDs, and stop reasons recorded in [`per_prompt_results.jsonl`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/evaluation/per_prompt_results.jsonl).

---

### 6. Verification of Actual SFT Training Behavior

Through synthetic tensor tests reflecting `chakrview/training/sft_engine.py`:
1. **Causal Target Alignment:** Verified that input position $i$ predicts target position $i$, aligning position $len(p\_toks)-1$ to predict $r\_toks[0]$ and position $len(input\_seq)-1$ to predict EOS.
2. **Assistant-Only Target Masking:** Verified prompt token positions in target tensor are masked with `-100`.
3. **EOS Supervision:** Verified EOS token ID (1) is explicitly supervised as the final target token.
4. **All-Masked Target Handling:** Proved that `torch.nn.functional.cross_entropy` with `ignore_index=-100` returns `NaN` when all target positions are masked, confirming why batch active token counts must be checked.
5. **Finite Loss and Gradients:** Proved forward pass on minimal valid batch produces finite scalar loss and valid parameter gradients.

---

### 7. Claim-by-Claim Audit of Batch 37

| Claim ID | Batch 37 Claim | Sprint Finding | Verdict | Evidence |
|---|---|---|:---:|---|
| `CLM-B37-01` | "12 of 19 records are genuinely supported" | Batch 37 used a 3-word overlap heuristic. Stricter separation reveals only 11 records are supported, 5 are partial, and 3 are unsupported (including UDP buffer attribution for decoding bandwidth). | **QUALIFIED** | [`record_validation.json`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/audits/record_validation.json) |
| `CLM-B37-02` | "`EXECUTION_PASS` demonstrates actual execution" | Executed function definitions only without invocation. Behavioral validation requires actual execution on inputs. | **REJECTED** | [`SafeCodeExecutionHarness`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/audits/validator.py) |
| `CLM-B37-03` | "Baseline independently reproduced with dynamic metadata" | 0.0% first-token EOS rate confirmed, but checkpoint hash and EOS ID were hardcoded strings in Batch 37. | **QUALIFIED** | [`run_manifest.json`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/evaluation/run_manifest.json) |
| `CLM-B37-04` | "Claimed source inventory covers all files in dataset" | Verified: all 16 referenced files exist, and all 27 corpus files match recorded SHA-256 digests. | **CONFIRMED** | [`source_inventory.json`](file:///d:/Project/ChakrView/experiments/batch37_independent_audit/audits/source_inventory.json) |
| `CLM-B37-05` | "Reported category count matches actual manifest" | Verified: `manifest.json` defines exactly 9 categories covering all 27 corpus files. | **CONFIRMED** | [`manifest.json`](file:///d:/Project/ChakrView/data/raw/manifest.json) |
| `CLM-B37-06` | "500-example expansion plan supported by source corpus" | Corpus size is sufficient (1.57 MB, >10k lines), but automated paragraph extraction tooling does not yet exist. Roadmap is feasible but unverified. | **QUALIFIED** | [`expansion_plan.json`](file:///d:/Project/ChakrView/experiments/batch37_independent_audit/data/expansion_plan.json) |

---

### 8. Enforced SFT Readiness Gate

Machine-readable gate output ([`sft_readiness.json`](file:///d:/Project/ChakrView/experiments/batch38_audit_hardening/gates/sft_readiness.json)):
- **`DATASET_READY`:** **`false`**
- **`PIPELINE_READY`:** **`true`**
- **`BASELINE_READY`:** **`true`**
- **`SFT_READY`:** **`false`**
- **Overall Decision:** **`BLOCKED`**

#### Blocking Reasons:
1. **Record Count Below Overfitting Threshold:** Dataset has only 19 records. Prior experiments (Batch 29/34) proved that training on <40 records for $\ge 5$ epochs causes 70–80% rank-1 premature EOS collapse. Required evidence: expansion to $\ge 500$ verified records.
2. **Unsupported Claims in Current Records:** 3 records (`EXP-TR-EN-02`, `EXP-TR-EN-03`, `EXP-TR-MT-01`) make claims not grounded in cited raw corpus text. Required evidence: realign records with grounded corpus passages.

#### Outstanding Semantic & Human Review Requirements:
- Independent human review is pending for all records (`human_review_status='NOT_VERIFIED'`).

---

### 9. Test Suite Execution Results

Executed regression suite:
```text
tests\test_pretraining_engine.py ......                                  [ 10%]
tests\test_batch28_stage_d_gate.py ....                                  [ 17%]
tests\test_sft_engine.py ...                                             [ 23%]
tests\test_batch31_chat_runtime.py ......                                [ 33%]
tests\test_batch32_acceptance.py ........                                [ 48%]
tests\test_batch33_integrity.py ...                                      [ 53%]
tests\test_batch34_sft_diagnostics.py ...                                [ 58%]
tests\test_batch35_recovery.py ...                                       [ 64%]
tests\test_batch36_readiness.py ...                                      [ 69%]
tests\test_batch37_independent_audit.py ....                             [ 76%]
tests\test_batch38_audit_hardening.py .............                      [100%]

============================= 56 passed in 11.06s =============================
```
- **Exit Code:** 0
- **Total Tests Passed:** 56 / 56

---

### 10. Remaining Uncertainties & Smallest Justified Next Step

#### Remaining Uncertainties:
1. Automated passage extraction tooling has not yet been implemented to extract 500 high-quality instruction pairs from the 27 raw corpus files.
2. Independent human evaluation of multilingual naturalness (Hindi and Hinglish) remains outstanding.

#### Smallest Justified Next Step:
Construct an automated, length-regularized passage extraction tool that scans the 7 large Stage B raw files (`stage_b_algorithms.txt`, `stage_b_systems.txt`, etc.) and pairs natural question templates with grounded paragraphs, generating an expanded dataset candidate for human review and readiness evaluation.
