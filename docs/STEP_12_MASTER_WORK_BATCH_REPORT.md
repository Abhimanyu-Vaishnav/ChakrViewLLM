# Master Work Batch 12 — Benchmark Validity, Failure Attribution & Training Experiment Readiness Report

**Repository:** `D:\Project\ChakrView`  
**Execution Context:** Step 12 Evaluator Validity & Error Attribution Milestone  
**Baseline Commit:** `a16c58a` (`feat(bench): strengthen neural capability evaluation`)  
**Branch:** `main` (Ahead of origin/main: 35)  
**Safety & Governance Status:** `TRAINING READINESS: NOT READY FOR TRAINING` ($\Delta W = 0$)

---

## 1. Executive Baseline & Repository Audit

### 1.1 Repository State
- **Baseline Git HEAD:** `a16c58a` verified.
- **Tracked Assets:** Evaluated benchmark harness, v2.0 prompt suite, and all six short and long result artifacts in `evaluation/results/v2/`.
- **Working Tree Integrity:** All checkpoints, runtime caches, and the official review queue (`qa_review_queue_v0.3.0.jsonl`, 30/30 records `UNREVIEWED`) remain untouched.

### 1.2 Verification of Protected Artifacts ($\Delta W = 0$)
All four protected artifact hashes were verified bit-exact via automated SHA-256 digests:

| Artifact Description | File Path | Expected SHA-256 Digest | Observed SHA-256 Digest | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Canonical Random Model Weights** | In-Memory (Seed 42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH (Canonical)** |
| **Stage C 9k Checkpoint** | `checkpoint_0009000.pt` | `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23` | `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23` | **MATCH (Frozen)** |
| **Stage D 500-Step Checkpoint** | `checkpoint_0000500.pt` | `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449` | `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449` | **MATCH (Frozen)** |
| **QA Pilot Manifest** | `qa_pilot_manifest.json` | `b7ebf80cd273e9cbbb9f60311ffeda34bda720a46d5f71ae8f8f6a067fb90604` | `b7ebf80cd273e9cbbb9f60311ffeda34bda720a46d5f71ae8f8f6a067fb90604` | **MATCH (Frozen)** |

---

## 2. Evaluator Validity Audit & Hardening (v2.0 $\to$ v2.1)

### 2.1 Defects Identified in v2.0 Evaluator
During independent inspection of `evaluate_deterministic_match()` in `evaluation/neural_benchmark_harness.py`, three specific structural vulnerabilities were discovered:
1. **Unbounded Substring Matching (False Positive Risk):**
   - *Issue:* In v2.0, substring checks used `k.lower() in gen_lower`. This would evaluate to `True` if an expected number appeared inside an unrelated larger number (e.g., target `'27'` matching inside `'127'` or `'270'`), or an entity like `'Corbett'` matching inside an unrelated token.
   - *Impact on Model Scores:* We verified that across all existing model outputs, this defect did not inflate scores because neither Stage C nor Stage D generated numbers containing target answers. However, it presented an unacceptable soundness defect for future evaluations.
2. **Superficial Code Syntax Verification (Syntax vs. Functional Risk):**
   - *Issue:* In v2.0, `deterministic_syntax_check` only checked whether target substrings like `"def is_positive"` and `"> 0"` appeared in the output text. A completely broken Python syntax string (e.g. `def is_positive(x) return x > 0` missing the colon) would pass.
   - *Fix:* Added actual Python Abstract Syntax Tree (`ast.parse()`) validation. The evaluator now verifies that Python snippets compile into a valid syntax tree, and then verifies that required architectural components are present.
3. **Empty Output & Boundary Handling:**
   - *Issue:* Empty or whitespace-only generations could produce undefined behavior or spurious matches if empty keys existed. The evaluator now explicitly fails-closed on empty generations.

### 2.2 Evaluator Version 2.1.0 Specification
- **Version Identifier:** `EVALUATOR_VERSION = "2.1.0"`
- **Regex Word Boundaries:** All numeric and text substring keys are evaluated using `(?<!\w)re.escape(k)(?!\w)`.
- **AST Parsing for Python:** Python code outputs must pass `ast.parse()` without throwing `SyntaxError`.
- **Zero Score Drift on Historical Runs:** We re-evaluated all six frozen model benchmark runs under v2.1.0; exactly $0/20$ tasks passed across all runs. The $0.0\%$ accuracy baseline is robust and not an artifact of evaluator bugs.

---

## 3. Prompt Independence & Provenance Audit

We audited all 30 prompts in `evaluation/benchmark_prompts_v2.0.json`:
1. **Schema & ID Uniqueness:** All 30 prompt IDs are globally unique, with non-empty fields for prompt, provenance, expected answer, and human rubric.
2. **Domain Coverage:** Exactly 5 prompts across each of the 6 canonical domains:
   - `passage_factual_recall`
   - `passage_explanation_summarization`
   - `arithmetic_logic`
   - `algorithmic_reasoning`
   - `safe_coding_structure`
   - `open_continuation`
3. **Passage Support Invariant:** Every question in `passage_factual_recall` and `passage_explanation_summarization` contains an explicit source passage preceding the question. In all 10 passage prompts, the target answer is directly attested by the text.
4. **Independent Arithmetic Verification:**
   - `12 + 15 = 27` (Attested)
   - `50 - 18 = 32` (Attested)
   - `8 * 7 = 56` (Attested)
   - `72 / 9 = 8` (Attested)
   - `24 - 6 = 18` (Attested)
5. **Evaluation Boundaries:**
   - 20 tasks are objectively checkable via deterministic rules.
   - 10 tasks (summarization and open continuation) are strictly evaluated via explicit human-review rubrics. We do not use automated LLM self-evaluation or proxy models.

---

## 4. Auditable Model Failure Attribution & Error Analysis

Using the audited v2.1 evaluator, every generation from the frozen models was systematically categorized into an empirical failure taxonomy:

### 4.1 Failure Taxonomy Definitions
- **`degeneration / character_loop`:** Model outputs repetitive single characters or punctuation without word formation.
- **`repetition / loop_collapse`:** Model enters an autoregressive n-gram loop (e.g. repeating phrases).
- **`template_leakage / memorization`:** Model outputs memorized scratchpad text from narrow training data regardless of prompt context.
- **`arithmetic_error / ungrounded`:** Model fails to perform calculation or outputs unrelated numbers.
- **`instruction_noncompliance`:** Model ignores format/question constraints.
- **`syntax_error / non_code`:** Code prompt output is non-executable or syntactically invalid.
- **`unsupported_claim / non_grounded`:** Response claims information absent from the supplied passage.

### 4.2 Aggregate Failure Breakdown Across Models (Short 32-Token Horizon)

| Failure Category | Random Baseline (Seed 42) | Stage C 9k Checkpoint | Stage D 500 Checkpoint |
| :--- | :---: | :---: | :---: |
| **Degeneration / Character Loop** | 30 / 30 (100%) | 0 / 30 (0%) | 0 / 30 (0%) |
| **Repetition / Loop Collapse** | 0 / 30 (0%) | 11 / 30 (36.7%) | 1 / 30 (3.3%) |
| **Template Leakage / Memorization** | 0 / 30 (0%) | 0 / 30 (0%) | 24 / 30 (80.0%) |
| **Arithmetic / Calculation Failure** | 0 / 30 (0%) | 5 / 30 (16.7%) | 0 / 30 (0%)* |
| **Instruction Noncompliance** | 0 / 30 (0%) | 5 / 30 (16.7%) | 0 / 30 (0%)* |
| **Syntax Error / Non-Code** | 0 / 30 (0%) | 3 / 30 (10.0%) | 2 / 30 (6.7%) |
| **Unsupported Claim / Non-Grounded** | 0 / 30 (0%) | 3 / 30 (10.0%) | 1 / 30 (3.3%) |
| **Irrelevant / Off-Topic Response** | 0 / 30 (0%) | 3 / 30 (10.0%) | 2 / 30 (6.7%) |

*\*Note: For Stage D, template leakage supersedes arithmetic/instruction errors because the model unconditionally emitted canned scratchpad text (`<<2*2=200>>12`) across all domains.*

### 4.3 Prompt-by-Prompt Comparison Table (Representative Sample)

| Prompt ID | Domain | Expected Answer | Stage C Output (First 40 chars) | Stage C Failure Mode | Stage D Output (First 40 chars) | Stage D Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PASS-FACT-01` | Passage Recall | `1921` | `The Great of the Great of the Great...` | `repetition / loop_collapse` | `The answer is 200.` | `template_leakage` |
| `PASS-FACT-02` | Passage Recall | `Jim Corbett` | `The Great of the Great of the Great...` | `repetition / loop_collapse` | `The answer is 200.` | `template_leakage` |
| `PASS-FACT-03` | Passage Recall | `Aryabhata` | `The first of the first of the first...` | `repetition / loop_collapse` | `The answer is 200.` | `template_leakage` |
| `PASS-FACT-04` | Passage Recall | `chloroplasts` | `The first of the group of the group...` | `repetition / loop_collapse` | `So he has 200*2=<<2*2=200>>12 years old.` | `template_leakage` |
| `PASS-FACT-05` | Passage Recall | `64 bytes` | `The famous is the place of the group...` | `repetition / loop_collapse` | `The number of the number of the number...` | `repetition / loop_collapse` |
| `ARITH-01` | Arithmetic | `27` | `The number of the government of the...` | `arithmetic_error` | `240>>40\nThere are 50*2=<<200*2=60>>40` | `template_leakage` |
| `ARITH-02` | Arithmetic | `32` | `The first of the place of the place...` | `arithmetic_error` | `<<240=240>>400\nThere are 500/2=<<200*2=6` | `template_leakage` |
| `LOGIC-01` | Logic | `Yes` | `The first of the group of the group...` | `instruction_noncompliance` | `200>=<<2*2=200>>120 cans\nSo he has 200*2` | `template_leakage` |
| `CODE-01` | Code Structure | `def is_positive` | `The first of the group of the group...` | `syntax_error / non_code` | `2=<<2*2=200>>120\nSo he has 200*2=<<2*2=2` | `syntax_error / template_leakage` |
| `CONT-01` | Continuation | `isolation` | `The first of the group of the group...` | `repetition / loop_collapse` | `. There are 5 of the number of pakes is` | `template_leakage` |

---

## 5. Controlled Repeatability Verification

- Re-executed repeated greedy inference passes ($T=0.0$) using distinct `NeuralBenchmarkHarness` instances on CPU.
- Generated token ID streams across repeated runs are bit-for-bit identical ($T_1 == T_2$).
- Verified that cold-start warmups are completely isolated from timing measurements.

---

## 6. Future Training Experiment Preparation (Design Only)

We prepared an auditable research design in [`docs/experiments/EXP_SPEC_REPLAY_REGULARIZED_STAGE_D.md`](file:///d:/Project/ChakrView/docs/experiments/EXP_SPEC_REPLAY_REGULARIZED_STAGE_D.md):
- **Core Hypothesis:** Joint training of student weights on reviewed QA data while anchoring logits to the permanently frozen Stage C 9k model ($D_{\text{KL}}$ anchor loss) and mixing $20\%$ general pretraining replay tokens will eliminate template mode collapse and prevent catastrophic forgetting.
- **Ablation Matrix:** Includes unregularized control (`RUN-A0`), replay-only control (`RUN-A1`), anchor-only control (`RUN-A2`), and full multi-task pilot (`RUN-EXP`).
- **Admission Preconditions:** Training readiness remains strictly `NOT READY FOR TRAINING`. No training may proceed until the Founder explicitly reviews the 30 candidate QA records and signs off on the dataset manifest.

---

## 7. Regression Testing & Execution Evidence

1. **`tests/test_neural_benchmark_harness.py`**:
   - `test_01_synthetic_metric_calculations`: PASSED
   - `test_02_harness_fails_closed_on_missing_checkpoint`: PASSED
   - `test_03_random_baseline_benchmark_execution`: PASSED
   - `test_04_save_benchmark_results_serialization`: PASSED
   - `test_05_prompt_suite_v2_schema_and_domain_coverage`: PASSED
   - `test_06_checkpoint_identity_and_hash_resolution`: PASSED
   - `test_07_deterministic_match_evaluation`: PASSED (Includes word boundary, AST syntax, and failure edge cases)
2. **`tests/test_qa_review_tooling.py` & `tests/test_step103_consolidation.py`**:
   - Clean checkout availability, 100% UNREVIEWED queue invariant, split isolation, non-mutation: ALL 12 TESTS PASSED.
3. **Total Test Suite:** 19/19 tests passing with Exit Code 0.

---

## 8. Final Training-Readiness Verdict

$$\mathbf{TRAINING\ READINESS:\ NOT\ READY\ FOR\ TRAINING}$$

- **Justification:** While evaluation infrastructure and research specifications are now fully rigorous and validated, the Founder Review Gate for the Stage D QA dataset remains pending human decision-making. No training, optimizer step, or weight mutation was executed ($\Delta W = 0$).

---

## 9. Next Research & Engineering Priorities

1. **Founder Decision Phase:** Complete semantic review of the 30 QA candidates in `checkpoints/experiments/neural_learning_v02/stage_d_design/qa_review_queue_v0.3.0.jsonl` using the tracked workflow tool.
2. **Experiment Execution (Post-Gate):** Once authorized, execute the controlled 4-way ablation experiment defined in `EXP_SPEC_REPLAY_REGULARIZED_STAGE_D.md`.
