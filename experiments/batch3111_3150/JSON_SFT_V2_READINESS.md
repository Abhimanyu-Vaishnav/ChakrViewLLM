# ChakrView Batch 3111–3150: JSON SFT V2 — Readiness Report

## Executive Summary & Final Verdict

**FINAL VERDICT:** `READY FOR EXPLICIT TRAINING AUTHORIZATION`

This engineering report documents the comprehensive failure diagnosis of the Batch 3071–3110 JSON SFT pilot, the mathematical and token-level characterization of its repetitive degeneration, the principled redesign into a multi-instance controlled SFT dataset (Batch 3111–3150), complete data-split and schema isolation, automated test verification, and frozen preflight gates.

Zero optimizer steps, training passes, or background fine-tuning processes were executed during this preparation batch. Canonical checkpoints, canonical tokenizer artifacts, and `chakrview/runtime/session.py` remain untouched and bit-identical.

---

## 1. Verified Findings & Diagnosis of the Previous Pilot (Batch 3071–3110)

### Empirical Baseline & Outcome Facts
1. **Candidate Identity & Parameters:**
   - Candidate checkpoint: `experiments/batch3071_3110/json_sft_pilot_run/sft_best_checkpoint.pt` (SHA-256: `1a812fc838eb1fd1cfe49e269a01aed9fe9f4c0f21b53585b620d2bc0d3f27e2`).
   - Architecture: $V=4096$, $d=192$, $L=6$, $H=6$, $d_{\text{ffn}}=512$, context length 512.
   - Tied weights (`tok_emb` and `lm_head`): **3,443,136 unique trainable parameters** (4,229,568 raw `state_dict` floats).
2. **Structural Shift vs Syntactic Failure:**
   - The Step-100 candidate generated opening JSON syntax `{"` on **20/20 (100%)** held-out test prompts (compared to 0/20 on baseline).
   - The candidate produced **0/20 (0.0%)** syntactically valid JSON outputs.
   - The regression guard on the 4 designated raw-continuation prompts passed with **0/4 immediate `<EOS>` (100% preserved)**.

### Token-Level Failure Analysis ([diagnose_token_failure.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/diagnose_token_failure.py))
Through deterministic logit and token tracing on the saved candidate weights, the exact failure mechanics were revealed:
1. **Format Induction at Boundary:**
   - At step 0 after `Response:`, top logit is token `126` (`{`, logit +8.47).
   - At step 1, top logit is token `37` (`"`, logit +10.37).
2. **Key-Pair Entropy Collapse & Subword Attractors:**
   - In Batch 3071–3110, the 140 training examples contained 271 unique keys across ~135 distinct domains (~1.03 examples per key-pair).
   - In a 3.44M parameter model, presenting 135 distinct key pairs only once or twice does not establish consistent associative pathways between schema instructions and specific JSON key tokens.
   - Instead of emitting the requested keys, greedy decoding collapsed into the highest-frequency shared subword bigrams (`"st"` [304] $\to$ `"ri"` [357] $\to$ `"p"` [115] $\to$ `"_"` [98] $\to$ `"to"` [325] $\to$ `"_"` [98] $\to$ `"to"` [325]...).
   - 17/20 test cases fell into the `'{"strip_to_to_to_...'` loop; 3/20 fell into the `'{"stript", "strict", "strict"...'` loop.
   - The model never closed the key string quote or key-value colon, hitting the 48-token limit with unterminated strings.
3. **Confirmed vs Unproven:**
   - *Confirmed Fact:* Model learned `Response: -> {"` with high confidence.
   - *Confirmed Fact:* Model degenerated into an infinite subword loop before emitting closing braces.
   - *Confirmed Fact:* 1.03 examples per key pair is insufficient for 3.44M parameters to associate unseen instruction schemas with key generation.
   - *Unproven Hypothesis:* That a 3.44M parameter model is inherently incapable of generating complete JSON dictionaries. (To test this hypothesis, the model must be trained on multiple instances per schema).

---

## 2. Phase C & D: Learning Objective Separation & Dataset V2 Design

To isolate syntax completion from zero-shot schema generalization, Batch 3111–3150 separates the evaluation into three distinct objectives:
- **Capability A (Syntax Completion):** Producing complete, parseable JSON dictionaries `{ ... }` ending with `<EOS>`.
- **Capability B (Schema Adherence):** Producing exactly the requested keys and value types on familiar schemas.
- **Capability C (Schema Generalization):** Producing structured JSON for novel keys not seen during training.

### Multi-Instance Structural Training Design
Instead of 135 diffuse key pairs, training focuses on **4 core structural schemas** with **35 diverse instances per schema** (140 total structured examples):
1. **Server Status (`status`, `code`):** String state + integer HTTP status (`{"status": "ready", "code": 200}`).
2. **Network Metric (`metric`, `value`):** String telemetry name + floating-point measurement (`{"metric": "cpu_usage", "value": 42.5}`).
3. **Feature Flag (`feature`, `enabled`):** String feature name + boolean flag (`{"feature": "dark_mode", "enabled": true}`).
4. **Inventory Item (`name`, `count`):** String item name + integer count (`{"name": "server_rack", "count": 12}`).

Each schema contains varied values, realistic domain entities, and 3 distinct prompt templates.

### Dataset Split Lineage & Composition ([DATASET_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/DATASET_MANIFEST.json))
- **Total Records:** 264
- **Train Split (`train.jsonl`):** 200 records
  * 140 Structured JSON (4 core schemas $\times$ 35 instances)
  * 60 Continuation Replay (from `corpus_english_foundation` and `corpus_stage_b_systems`)
- **Validation Split (`val.jsonl`):** 30 records
  * 20 Structured JSON (4 core schemas $\times$ 5 held-out instances with distinct prompt phrasing)
  * 10 Continuation Replay (from `corpus_english_systems` and `corpus_code_python`)
- **Held-Out Test Split (`test.jsonl`):** 34 records
  * 12 Familiar-Schema JSON (4 core schemas $\times$ 3 held-out instances with distinct test phrasing)
  * 12 Genuinely Unseen-Schema JSON (12 novel key pairs, e.g. `["pilot_callsign", "heading_deg"]`, `["telescope_filter", "exposure_sec"]`)
  * 10 Continuation Replay (from `corpus_code_foundation` and `corpus_math_foundation`)

### Anti-Leakage & Isolation Verification
1. **Zero Prompt Overlap:**
   - Split-specific prompt templates guarantee 0 prompt collisions across train, val, and test (`train_val = 0`, `train_test = 0`, `val_test = 0`).
2. **Zero Unseen Key Leakage:**
   - All 12 unseen test schema key-pairs are verified 100% absent from train and validation.
3. **Document-Level Replay Isolation:**
   - Replay sources are strictly partitioned at corpus document boundaries. Zero document overlap exists across splits (`train_val = 0`, `train_test = 0`, `val_test = 0`).
4. **Zero Historical Contamination:**
   - Test replay prompts are filtered against all historical evaluation sets (`batch3071`, `batch3021`, and `GENERATION_BENCHMARK.json`).

---

## 3. Training & Preflight Configuration

The proposed configuration is frozen in [SFT_EXPERIMENT_CONFIG.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/SFT_EXPERIMENT_CONFIG.json):

- **Execution Device:** CPU-only (`torch.device('cpu')`).
- **Base Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256 verified).
- **Canonical Tokenizer:** Vocab 4096 (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- **Training Examples:** 200 (`train.jsonl`).
- **Batch Size:** 4.
- **Pilot Steps:** 100 optimizer steps ($100 \times 4 = 400$ presentations = exactly $2.0$ passes over the 200 training examples).
- **Learning Rate:** $3 \times 10^{-5}$ (cosine decayed to $3 \times 10^{-6}$, `weight_decay=0.01`, `grad_clip=1.0`).
- **Validation Interval:** Every 10 steps using exclusively `val.jsonl`.
- **Checkpoint Saving:** Every 20 steps, plus atomic best-checkpoint saving (`sft_best_checkpoint.pt`) based on validation loss.
- **Output Directory:** Exclusively `experiments/batch3111_3150/json_sft_v2_pilot_run/`.

### Preflight Gate Verification ([run_preflight.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/run_preflight.py))
All 8 fail-closed gates executed cleanly with status `PASS`:
1. `parent_checkpoint`: PASS (Canonical Step-10,000 SHA-256 matches `96357705...`)
2. `tokenizer`: PASS (Vocab size 4096 verified)
3. `dataset_manifest_and_balance`: PASS (Train=200 [140/60], Val=30 [20/10], Test=34 [12 fam / 12 uns / 10 rep], Leakage=0, Doc-Isolated=True)
4. `label_masking_and_eos_policy`: PASS (Prompt tokens masked `-100`, instruction ends with EOS, replay unforced)
5. `configuration_reconciliation`: PASS (100 steps, batch size 4, lr 3e-5, seed 42)
6. `output_isolation`: PASS (`experiments/batch3111_3150/json_sft_v2_pilot_run`)
7. `best_checkpoint_saving_support`: PASS (`save_checkpoint` supports `is_best`)
8. `test_set_isolation`: PASS (`test.jsonl` strictly excluded from trainer invocation)

Preflight report recorded in [FINAL_PREFLIGHT_REPORT.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/FINAL_PREFLIGHT_REPORT.json).

---

## 4. Frozen Evaluation Protocol & Acceptance Criteria

The evaluation protocol is implemented in [run_evaluation.py](file:///d:/Project/ChakrView/experiments/batch3111_3150/run_evaluation.py).

### Baseline Frozen Results on V2 Test Set
Evaluated on canonical Step-10,000 checkpoint and frozen in [BASELINE_AND_CANDIDATE_EVALUATION.json](file:///d:/Project/ChakrView/experiments/batch3111_3150/BASELINE_AND_CANDIDATE_EVALUATION.json):
- **Familiar-Schema Test (12 cases):**
  * Valid JSON syntax: **0/12 (0.0%)**
  * Schema match: **0/12 (0.0%)**
- **Unseen-Schema Test (12 cases):**
  * Valid JSON syntax: **0/12 (0.0%)**
  * Schema match: **0/12 (0.0%)**
- **Replay Continuation (10 cases):**
  * Average tokens: **48.0**
  * Immediate `<EOS>`: **0/10**
- **Regression Guard (4 raw-continuation benchmark prompts):**
  * Immediate `<EOS>`: **0/4 (PASS)**

### Predeclared Acceptance Criteria
1. **Primary Success Criterion (Syntax Completion on Familiar Schemas):**
   - At least **50% ($\ge 6/12$)** of held-out familiar-schema prompts must produce syntactically valid, parseable JSON dictionaries (`json.loads` returning a `dict`).
2. **Secondary Criterion (Schema Adherence):**
   - Exact key-set match and field-value correctness reported separately.
3. **Generalization Metric (Unseen Schemas):**
   - Valid JSON syntax and schema match on the 12 genuinely unseen schemas reported independently to measure zero-shot structural generalization.
4. **Regression Guard Criterion:**
   - Exactly **0 immediate-EOS outputs (0/4)** on the 4 designated raw-continuation prompts.
   - If the regression guard fails, the run will be classified as a **regression failure** regardless of JSON performance.

---

## 5. Automated Test Suite Evidence

Automated pytest tests covering dataset construction, split isolation, causal contracts, fail-closed preflight, and parser metrics were executed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_json_sft_v2_batch3111_3150.py -v
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

Combined execution across Batch 3071 and Batch 3111 test suites: **16/16 PASSED (100%)**.

---

## 6. Execution Command for Authorized Training

Once separate explicit authorization is granted by the operator, the 100-step CPU training experiment can be executed with:

```powershell
.\.venv\Scripts\python.exe experiments/batch3111_3150/run_trainer.py --execute-authorized-training
```

After training completes, evaluation is executed with:

```powershell
.\.venv\Scripts\python.exe experiments/batch3111_3150/run_evaluation.py --candidate-path experiments/batch3111_3150/json_sft_v2_pilot_run/sft_best_checkpoint.pt
```

---

## 7. Safety Invariant Audit

- `chakrview/runtime/session.py`: Untouched and unmodified.
- Unstaged user modifications: 100% preserved.
- Canonical Checkpoint Step-10,000 SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-identical).
- Canonical Tokenizer: Vocab size 4096 (Verified bit-identical).
- Training Executed in this Turn: **0 optimizer steps**.
- Git Working Tree: No commits, stashes, or resets performed.
