# ChakrView Batch 3071–3110: JSON Capability Experiment V1 — Readiness Report

## Executive Summary & Final Verdict

**FINAL VERDICT:** `READY FOR EXPLICIT TRAINING AUTHORIZATION`

This engineering report documents the diagnosis of earlier multi-task SFT runs, the principled design and implementation of a focused JSON capability experiment (Batch 3071–3110), complete data-split isolation, causal loss contracts, automated test verification, and frozen preflight gates.

Zero optimizer steps, training passes, or background fine-tuning processes were executed during this preparation batch. Canonical checkpoints, canonical tokenizer artifacts, and `chakrview/runtime/session.py` remain untouched and bit-identical.

---

## 1. Verified Diagnosis Facts from Previous Experiments

1. **Canonical Baseline Architecture & Unique Trainable Parameters:**
   - Architecture: $V=4096$, $d_{\text{model}}=192$, $L=6$, $H=6$, $d_{\text{ffn}}=512$, context length 512.
   - Tied Weights: The embedding layer `tok_emb` and LM head `lm_head` share identical weight tensors ($4096 \times 192 = 786,432$ parameters).
   - Parameter Accounting:
     * Raw `state_dict` element count: **4,229,568** floats.
     * Unique trainable parameters: $4,229,568 - 786,432 = \mathbf{3,443,136}$.
   - Parent Checkpoint: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` verified with exact SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.

2. **Step-150 Candidate Identity:**
   - `experiments/batch3021_3060/balanced_sft_pilot_run/sft_best_checkpoint.pt` was confirmed to be bit-identical to the periodic Step-150 checkpoint (`sft_checkpoint_step_150.pt`) with validation loss $4.4541$ (lowest across 160 steps).

3. **Loss Dynamics vs. Capability Divergence (Root Cause Analysis):**
   - *Directly Measured Fact:* In Batch 3021–3060, training loss decreased from $7.85 \to 3.03$ and validation loss decreased from $7.91 \to 4.45$. However, held-out exact match remained $0/28$ ($0.0\%$), and JSON syntax validity remained $0/4$ ($0.0\%$).
   - *Causal Implementation Audit:* Causal target alignment, prompt loss masking (`-100` on prompt tokens), and EOS supervision were verified to behave strictly as intended in `chakrview/training/prompt_contract.py` and `sft_engine.py`.
   - *Natural Continuation Regression Resumed:* In Batch 2821–2860 (pure instruction tuning), $15/28$ raw-completion prompts collapsed to immediate `<EOS>`. In Batch 3021–3060, the $30\%$ unconditioned replay ratio completely cured this regression, yielding $0/4$ immediate `<EOS>` on the regression benchmark.
   - *Associative Capacity Collapse:* A 3.44M parameter model does not have sufficient capacity to simultaneously learn diverse lexical extraction, multi-sentence summarization, question answering, algebraic reasoning, and JSON synthesis across only 280 instruction examples. The validation loss reduction reflected the model learning dominant unigram/bigram token priors (e.g. outputting `"The "` with massive logit $+8.09$) rather than generalized instruction comprehension.

4. **Hypothesis for Batch 3071–3110:**
   - Narrowing the instruction objective strictly to a consistent, two-field JSON synthesis task with schema specifications while preserving the $30\%$ unconditioned replay ratio allows testing whether structured output syntax can emerge within a 3.44M parameter neural brain without triggering natural continuation regressions.

---

## 2. Dataset Lineage, Mixture & Split Isolation

### Mixture Composition
- **Total Examples:** 260
- **Mixture Ratio:** $70\%$ structured-output JSON instructions ($180$ total), $30\%$ natural continuation replay ($80$ total).
- **Split Breakdown:**
  * **Train Split (`train.jsonl`):** 200 records ($140$ JSON, $60$ replay).
  * **Validation Split (`val.jsonl`):** 30 records ($20$ JSON, $10$ replay).
  * **Held-Out Test Split (`test.jsonl`):** 30 records ($20$ JSON, $10$ replay).

### Strict Anti-Leakage & Isolation Guarantees
1. **Schema & Field Independence:**
   - Training schemas use 20 diverse, realistic two-field combinations (e.g., `["item_name", "price_usd"]`, `["sensor_id", "status"]`, `["function_name", "return_type"]`).
   - The held-out test split uses completely unseen key combinations (e.g., `["latitude", "longitude"]`, `["module_path", "line_count"]`, `["chemical_formula", "molecular_weight"]`).
   - Zero test prompt or test key pair appears in the training or validation splits.
2. **Replay Source Document Isolation:**
   - Raw passages from `corpus_code_foundation.txt` and `corpus_math_foundation.txt` were partitioned at document line boundaries.
   - Zero document overlap exists between train, val, and test splits (`train_val = 0`, `train_test = 0`, `val_test = 0`).
3. **Zero Benchmark Overlap:**
   - All replay prompts were filtered against all historical evaluation benchmarks (`batch3021_3060/test.jsonl`, `batch2821_2860`, `batch2781_2820/GENERATION_BENCHMARK.json`).
4. **Target Validity:**
   - All 180 instruction targets across train, val, and test parse as 100% syntactically valid JSON dictionaries.

---

## 3. Training & Preflight Configuration

The training pipeline is isolated in `experiments/batch3071_3110/run_trainer.py`, configured via `experiments/batch3071_3110/SFT_EXPERIMENT_CONFIG.json`:

- **Device:** CPU-only (`torch.device('cpu')`).
- **Base Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256 verified).
- **Tokenizer:** Canonical Vocab 4096 (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- **Training Examples:** 200 (`train.jsonl`).
- **Batch Size:** 4.
- **Pilot Steps:** 100 optimizer steps ($100 \times 4 = 400$ presentations = exactly $2.0$ passes over the dataset).
- **Learning Rate:** $3 \times 10^{-5}$ (cosine decayed to $3 \times 10^{-6}$).
- **Validation Interval:** Every 10 steps using exclusively `val.jsonl`.
- **Checkpoint Saving:** Every 20 steps, plus atomic best-checkpoint saving (`sft_best_checkpoint.pt`) based on validation loss.
- **Output Directory:** Exclusively `experiments/batch3071_3110/json_sft_pilot_run/`.

### Preflight Gates Enforcement (`run_preflight.py`)
All 8 fail-closed gates executed cleanly with status `PASS`:
1. `parent_checkpoint`: PASS (Canonical Step-10,000 SHA-256 matches `96357705...`)
2. `tokenizer`: PASS (Vocab size 4096 verified)
3. `dataset_manifest_and_balance`: PASS (Train=200 [140/60], Val=30 [20/10], Test=30 [20/10], Leakage=0, Doc-Isolated=True)
4. `label_masking_and_eos_policy`: PASS (Prompt tokens masked `-100`, instruction ends with EOS, replay unforced)
5. `configuration_reconciliation`: PASS (100 steps, batch size 4, lr 3e-5, seed 42)
6. `output_isolation`: PASS (`experiments/batch3071_3110/json_sft_pilot_run`)
7. `best_checkpoint_saving_support`: PASS (`save_checkpoint` supports `is_best`)
8. `test_set_isolation`: PASS (`test.jsonl` strictly excluded from trainer invocation)

---

## 4. Frozen Evaluation Protocol & Baseline Results

The evaluation protocol is implemented in `experiments/batch3071_3110/run_evaluation.py`. Baseline results on Step-10,000 were computed and frozen:

### Evaluation Metrics
- **Strict JSON Parsing:** Evaluated with `json.loads`. Responses must parse to a Python `dict`.
- **Schema Adherence:** Evaluates if parsed keys exactly match expected `target_keys` without extra or missing keys.
- **Regression Guard:** Evaluates immediate `<EOS>` and empty rate on the designated 4 natural-continuation prompts from `experiments/batch2781_2820/GENERATION_BENCHMARK.json`.

### Canonical Step-10,000 Baseline Frozen Results
- **Held-Out Test Set (20 JSON cases):**
  * Syntactically Valid JSON: **$0/20$ ($0.0\%$)**
  * Schema Match: **$0/20$ ($0.0\%$)**
  * Empty Output Count: **$0/20$ ($0.0\%$)**
- **Held-Out Replay (10 cases):**
  * Average Generated Tokens: **48.0**
  * Immediate `<EOS>`: **0**
- **Regression Guard (4 raw-continuation cases):**
  * Immediate `<EOS>`: **$0/4$ ($0.0\%$)** -> **PASS**
  * Empty Output: **$0/4$ ($0.0\%$)**

### Predeclared Acceptance Criteria for Candidate
1. **Primary Success:** At least $50\%$ ($10/20$) of held-out structured-output prompts must produce syntactically valid JSON dictionaries.
2. **Regression Guard:** Zero ($0/4$) immediate-EOS outputs on the designated 4 natural-continuation regression prompts.
3. **Classification Rule:** If the candidate produces $\ge 50\%$ valid JSON but triggers any immediate `<EOS>` on the regression guard, the run will be classified as a **regression failure**.

---

## 5. Automated Test Evidence

Automated pytest suites were run against the implementation:

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_json_sft_batch3071_3110.py -v
```

**Results:**
- `test_dataset_record_counts_and_splits`: **PASSED**
- `test_split_isolation_and_no_leakage`: **PASSED**
- `test_document_level_replay_isolation`: **PASSED**
- `test_json_target_syntax_validity`: **PASSED**
- `test_prompt_masking_and_causal_contract`: **PASSED**
- `test_preflight_gate_enforcement`: **PASSED**
- `test_trainer_held_out_exclusion`: **PASSED**
- `test_strict_json_parser_metrics`: **PASSED**

Combined test execution with prior isolation suite (`tests/test_final_isolation_batch3021_3060.py`): **12/12 PASSED (100%)**.

---

## 6. Execution Command for Authorized Training

Once separate explicit authorization is granted by the operator, the 100-step CPU training experiment can be executed with:

```powershell
.\.venv\Scripts\python.exe experiments/batch3071_3110/run_trainer.py --execute-authorized-training
```

After training completes, evaluation is executed with:

```powershell
.\.venv\Scripts\python.exe experiments/batch3071_3110/run_evaluation.py --candidate-path experiments/batch3071_3110/json_sft_pilot_run/sft_best_checkpoint.pt
```

---

## 7. Safety Invariant Audit

- `chakrview/runtime/session.py`: Untouched and unmodified.
- Unstaged user modifications: Preserved without modification or deletion.
- Canonical Checkpoint Step-10,000 SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-identical).
- Canonical Tokenizer: Vocab size 4096 (Verified bit-identical).
- Training Executed in this Turn: **0 optimizer steps**.
- Git Working Tree: No commits, stashes, or resets performed.
