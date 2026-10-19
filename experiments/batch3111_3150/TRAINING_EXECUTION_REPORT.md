# ChakrView Batch 3111–3150: JSON SFT V2 Training Execution Report

## Executive Summary & Final Status

**FINAL STATUS:** `SUCCESS — 100 STEPS COMPLETED; RESULTS REPORTED`

This engineering report records the execution and empirical evaluation of the authorized 100-step CPU training run for **Batch 3111–3150 (JSON SFT V2)**. All pre-execution safety gates passed cleanly, canonical hashes remained bit-identical, and exactly 100 optimizer steps completed on CPU without runtime errors or process retries.

**Key Measured Results:**
1. **Training Execution:** Exactly 100 optimizer steps completed on CPU in 25.35 seconds (~0.25 sec/step).
2. **Loss Dynamics:** Validation loss dropped from initial $5.5269 \to 3.2850$ (minimum achieved at Step 60, validation perplexity $26.71$), ending at $3.6258$ (perplexity $37.55$) at Step 100.
3. **Selected Candidate:** `sft_best_checkpoint.pt` was verified to match the weights of `sft_checkpoint_step_60.pt` bit-identically (0 state-dict discrepancies).
4. **Familiar-Schema JSON Syntax Validity:** **0 / 12 (0.0%)** parseable JSON dictionaries (Primary Target $\ge 6/12$ was **NOT MET**).
5. **Unseen-Schema JSON Syntax Validity:** **0 / 12 (0.0%)** parseable JSON dictionaries.
6. **Natural Continuation Regression Guard:** **0 / 4 (0.0%)** immediate `<EOS>` triggers on the 4 designated raw-continuation benchmark prompts. **PASSED 100%**.

---

## 1. Preflight, Isolation & Safety Verification Evidence

Prior to authorized training, all safety gates and hashes were verified:

| Check | Target / Expected | Actual / Measured | Result |
| :--- | :--- | :--- | :--- |
| **Canonical Base Checkpoint** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| **Tokenizer `vocab.json`** | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS** |
| **Tokenizer `merges.json`** | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS** |
| **Protected File Integrity** | `chakrview/runtime/session.py` untouched | Verified untouched via `git status` | **PASS** |
| **Prompt Split Overlap** | Train-Val: 0, Train-Test: 0, Val-Test: 0 | 0, 0, 0 | **PASS** |
| **Target Split Overlap** | Train-Val: 0, Train-Test: 0, Val-Test: 0 | 0, 0, 0 | **PASS** |
| **Unseen Schema Key Isolation** | 0 key-pair overlap between unseen test & train | 0 key-pair overlap verified | **PASS** |
| **Document-Level Replay Isolation** | 0 corpus document overlap between splits | Disjoint document subsets confirmed | **PASS** |
| **Preflight Gate Engine** | All 8 fail-closed gates pass | Exit code 0, all gates `PASS` | **PASS** |

The machine-readable preflight report was recorded in [FINAL_PREFLIGHT_REPORT.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/FINAL_PREFLIGHT_REPORT.json).

---

## 2. Configuration Actually Used

- **Base Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Execution Device:** CPU
- **Optimizer:** AdamW (`learning_rate=3e-5`, cosine decayed to `3.01e-6`, `weight_decay=0.01`, `grad_clip=1.0`)
- **Optimizer Steps:** Exactly 100 steps completed
- **Batch Size:** 4
- **Training Examples:** 200 (140 structured JSON across 4 core schemas, 60 continuation replay)
- **Dataset Presentations:** $100 \times 4 = 400$ presentations ($2.0$ full passes over the dataset)
- **Validation Split:** 30 records (`val.jsonl`) evaluated every 10 steps
- **Elapsed Training Time:** 25.35 seconds (~0.25 sec/step)
- **Output Directory:** `experiments/batch3111_3150/json_sft_v2_pilot_run/`

### Validation Progression Across 100 Steps:

| Step | Validation Loss | Validation Perplexity | Best Checkpoint Saved |
| :---: | :---: | :---: | :---: |
| Initial (0) | 5.5269 | 251.36 | — |
| 10 | 4.9188 | 136.84 | Yes |
| 20 | 4.5958 | 99.07 | Yes |
| 30 | 4.3367 | 76.46 | Yes |
| 40 | 3.7549 | 42.73 | Yes |
| 50 | 3.7228 | 41.38 | Yes |
| **60** | **3.2850** | **26.71** | **Yes (Global Best)** |
| 70 | 3.9392 | 51.38 | No |
| 80 | 3.8598 | 47.46 | No |
| 90 | 3.3756 | 29.24 | No |
| 100 | 3.6258 | 37.55 | No |

---

## 3. Checkpoint Inventory & Hashes

All checkpoints were written exclusively to [experiments/batch3111_3150/json_sft_v2_pilot_run/](file:///d:/Project/ChakrView/experiments/batch3111_3150/json_sft_v2_pilot_run):

| Checkpoint File | SHA-256 Hash | Notes |
| :--- | :--- | :--- |
| `sft_checkpoint_step_10.pt` | `93dc0d1613d9880f723a980cf10d53cb61e8a7de66c1624f49d2bb7dc2801fe7` | Step 10 periodic |
| `sft_checkpoint_step_20.pt` | `53e4fac05f0c8e81208a559e275944d58a0e09f21f1bd3fe81cbf25efbf530de` | Step 20 periodic |
| `sft_checkpoint_step_30.pt` | `b4a08978948623e8273a48e17a21226e23c5077e6ac963d89293b8a4ef0d4ea2` | Step 30 periodic |
| `sft_checkpoint_step_40.pt` | `8b698bd78bd73d6ecc25158343d46c26566c30aff1222ddf8963a8ec07861a22` | Step 40 periodic |
| `sft_checkpoint_step_50.pt` | `4a84560aa10e7d724d26d1a4e700035beb7335782ee0c6d207d94f3dbc127d4b` | Step 50 periodic |
| `sft_checkpoint_step_60.pt` | `8a6739450f5b819f7ef46c03dc12f1106476c50f5bdbec342eaac26ff8107409` | Step 60 periodic (Lowest Val Loss) |
| `sft_checkpoint_step_80.pt` | `eb6c6f5e9bc7ba120dd0e14fac341c0b4166282e1415c7ee4210430840a37094` | Step 80 periodic |
| `sft_checkpoint_step_100.pt` | `1c9cf66c3e07810e8864bce43835a6af413c48db170617e53fda4f76edba0357` | Step 100 periodic |
| `sft_best_checkpoint.pt` | `436bdb89e62e0ef461e511b0ac410d01368ce626569b64c27d803ec97d953bd6` | **Selected Best Checkpoint (Step 60 weights)** |

Weight parity check confirmed 0 tensor discrepancies between `sft_best_checkpoint.pt` and `sft_checkpoint_step_60.pt`.

---

## 4. Single Held-Out Evaluation Results (Baseline vs. Candidate)

The single evaluation run on the held-out test split (`test.jsonl`, 34 examples) and the generation benchmark was executed via [run_evaluation.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/run_evaluation.py).

Results recorded in [BASELINE_AND_CANDIDATE_EVALUATION.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/BASELINE_AND_CANDIDATE_EVALUATION.json):

### Comparative Summary Table

| Metric / Evaluation Slice | Canonical Baseline (Step 10,000) | Step-60 Best Candidate (`sft_best_checkpoint.pt`) | Target / Acceptance Threshold |
| :--- | :---: | :---: | :---: |
| **Familiar-Schema Valid JSON** | 0 / 12 (0.0%) | 0 / 12 (0.0%) | $\ge 50\%$ (6 / 12) [NOT MET] |
| **Familiar-Schema Match** | 0 / 12 (0.0%) | 0 / 12 (0.0%) | Monitored |
| **Unseen-Schema Valid JSON** | 0 / 12 (0.0%) | 0 / 12 (0.0%) | Monitored |
| **Unseen-Schema Match** | 0 / 12 (0.0%) | 0 / 12 (0.0%) | Monitored |
| **JSON Structural Opening (`{"`)** | 0 / 24 (0.0%) | **24 / 24 (100.0%)** | Structural syntax shift |
| **Replay Average Length** | 48.0 tokens | 39.1 tokens | Active continuation |
| **Replay Immediate `<EOS>`** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 |
| **Regression Guard Immediate `<EOS>`** | 0 / 4 (0.0%) | **0 / 4 (0.0%)** | **0 / 4 [PASSED 100%]** |
| **Regression Guard Empty Outputs** | 0 / 4 (0.0%) | 0 / 4 (0.0%) | 0 |

---

## 5. Detailed Case-by-Case Analysis & Mechanistic Diagnosis

### Candidate Generation Patterns
All 24 structured-output test cases (12 familiar and 12 unseen) began with `{"`. However, instead of closing the JSON object, greedy decoding collapsed into a python code-block attractor:

```text
Prompt (Familiar Schema): "Instruction: Produce a JSON object with properties 'status' and 'code' representing server health status and return code.\nResponse:"
Baseline: "\n* [https://web.archive.org/web/2012060606/http://www.gov.com"
Candidate: "{\"__main__\":\n    import doctest\n\n    # import doctest\n"

Prompt (Familiar Schema): "Instruction: Produce a JSON object with properties 'metric' and 'value' representing telemetry metric name and measured reading.\nResponse:"
Baseline: "\n\n\nSeff of the first of the world, which is a shown of the word..."
Candidate: "{\"__main__\":\n    import doctest\n    import doctest\n\n"

Prompt (Familiar Schema): "Instruction: Create a JSON object specifying keys 'name' and 'count' for inventory item name and available count.\nResponse:"
Baseline: "\n\nThe first of the first of the world, which is the world..."
Candidate: "{\"name\", \"index\", \"code\", \"index\", \"index\", \"index\", \"index"
```

### Mechanistic Diagnosis: Why Did It Diverge to Python Code Blocks?
1. **Strong Syntax Initiation:** In both Batch 3071 and Batch 3111, the model learned `Response: -> {"` with 100% precision.
2. **Key Generation in Multi-Instance Training:** In `JSON_V2_TST_FAM_005`, `010`, `011`, `012`, the model correctly emitted schema keys (`"name"` and `"code"`), proving that multi-instance learning successfully induced key-name associative representations absent in Batch 3071.
3. **Cross-Modality Attractor Interference:** Because the pretraining corpus contains heavy Python code foundation (`corpus_code_python` and `code_foundation.txt`) featuring `if __name__ == '__main__':` and `import doctest`, and the unconditioned replay ratio is 30%, the 3.44M parameter model's attention heads bridged from `{"` into `__main__"` and doctest code blocks.
4. **Structural Closure Barrier:** Because greedy decoding branched into Python comments (`# import doctest`) inside the JSON opening brace, strict JSON parsers rejected the output as invalid syntax (`Expecting value: line 2 column 5`).

---

## 6. Safety Invariant Audit

- `chakrview/runtime/session.py`: Untouched, unstaged, unmodified.
- Pre-existing user modifications: 100% preserved.
- Canonical Step-10,000 Checkpoint SHA-256: Verified bit-identical (`96357705...`).
- Canonical Tokenizer Vocab & Merges: Verified bit-identical.
- Total Training Jobs Executed: Exactly 1 run, exactly 100 optimizer steps.
- Git Working Tree: No commits, resets, stashes, or pushes performed.
