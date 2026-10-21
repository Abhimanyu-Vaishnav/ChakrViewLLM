# CHAKRVIEW v0.8: EXPERIMENT 02 RESEARCH & EVIDENCE REPORT
## String Copying & Learning Diagnostics: In-Context Value Extraction vs Mode Collapse

- **Author**: ChakrView Research & ML Engineering Agent
- **Execution Date**: 2026-10-21
- **Experiment Identifier**: `research_v08_json_sft_exp02_string_copy`
- **Output Directory**: `experiments/research_v08_json_sft/exp02_string_copy/`
- **Architecture**: ChakrMicro (3,443,136 unique parameters, 6 layers, $d_{\text{model}}=192$, $n_{\text{heads}}=6$, $d_{\text{ff}}=512$, context 512)
- **Primary Mission**: Diagnose Experiment 01 mode-collapse, establish parent baseline, and empirically evaluate whether ChakrMicro can learn single-field string extraction (copying an arbitrary in-context status string into structured JSON `{"status": "<val>"}`).

---

## 1. Experiment 01 Forensic Findings

Prior to designing Experiment 02, we conducted an empirical inspection of the actual implementation, training targets, tokenization, and checkpoint weights of Experiment 01 (`experiments/research_v08_json_sft/checkpoints/sft_best_checkpoint.pt`).

### Verified Forensic Diagnostics:
1. **Target Supervision & Alignment**:
   - `prompt_contract.py` and `sft_engine.py` were inspected and verified.
   - For instruction sequences, causal alignment is strictly valid:
     `input_seq: [BOS] + prompt_tokens + response_tokens`
     `target_seq: [-100]*len(prompt_tokens) + response_tokens + [EOS]`
   - The final prompt token (e.g. `Response:`) correctly aligns with the first response token `{`.
   - Both `"status"` and `"code"` fields were **consistently present** in 100% of the training labels (100/100 two-key records).
   - The status field was **not** un-supervised or masked out: all 18 response tokens (delimiters, status key, status value, code key, code value, braces) had active cross-entropy loss targets ($t \ne -100$).
2. **The Mechanism Behind `{"code": 3}`**:
   - Per-token loss and logit analysis on Record 0 (`{"status": "nominal", "code": 214}`):
     - At position 94 (after `{`), model predicts `"` with high confidence (loss = 0.09).
     - At position 95 (after `{"`), the ground truth is `stat`, but the model's top prediction is `co` (logit 5.98 vs `stat` logit 5.20).
     - Because `co` wins the argmax by 0.78 logits, greedy decoding branches immediately into `"code": `.
     - After `"code": `, position 107 predicts single-digit integer `3` (logit 8.25 vs next digit `4` logit 7.73).
     - After `3`, position 108 predicts `}` (logit 8.25), and position 109 predicts `<EOS>` (logit 11.83).
   - **Why did this happen?**
     - In the training set, status strings vary widely across 20 distinct multi-token strings (e.g. `['no', 'min', 'al']`, `['ac', 'ti', 've']`, `['de', 'gra', 'de', 'd']`), meaning the token immediately following `{"status": "` has high conditional entropy and per-token loss $\sim 5.72$.
     - Conversely, `"code": ` is followed by a compact numeric representation, and closing early with `}` and `<EOS>` produces a 6-token valid JSON object.
     - The cross-entropy objective averaged across the batch penalized the uncertain variable string tokens heavily during early steps, incentivizing the optimizer to settle into a structural attractor: emit the static syntax `{"code": 3}` which satisfies JSON validity and minimizes average token loss.
3. **Weight-Drift Metric Formulation**:
   - The weight-drift metric is formally defined as:
     $$\text{Drift} = \frac{\|\Delta W\|_2}{\|W_0\|_2} = \frac{\sqrt{\sum (\theta_t - \theta_0)^2}}{\sqrt{\sum \theta_0^2}}$$
   - At step 100 of Exp 01, relative drift was **1.69%**, remaining safely below the 4.0% boundary.

---

## 2. Lineage & Checkpoint Hashes

All code, data, and model artifacts were independently tracked and hashed (SHA-256):

| Artifact Description | Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Canonical Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Exp02 Frozen Train Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/train.jsonl` | `bf0da185d0a5172bffd422c0b64a2ecf664d532b56148c54b3ce5856d07f55c4` |
| **Exp02 Frozen Val Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/val.jsonl` | `d5750bff540387b4a4c8d68515fd9a1a25cbb5f2122645f5da0589f5c21fbb9c` |
| **Exp02 Frozen Test Dataset** | `experiments/research_v08_json_sft/exp02_string_copy/data/test.jsonl` | `b8ce9bfb6a1bfaba8ca7607a126b1babd33420839954d72a6af80e893f518f88` |
| **Exp02 Dataset Manifest** | `experiments/research_v08_json_sft/exp02_string_copy/data/dataset_manifest.json` | `d013e6290a245c2e611b032d01e22385071cbf084e3c176ef4c1fd3d05470f9c` |
| **Exp02 Trained Best Checkpoint** | `experiments/research_v08_json_sft/exp02_string_copy/checkpoints/exp02_best_checkpoint.pt` | `d31d2f9cc8be1a021b710a2f434db20d2f45e545e1c5fea10afcca1675f2bfe2` |
| **Exp02 Trained Final Step 120** | `experiments/research_v08_json_sft/exp02_string_copy/checkpoints/exp02_checkpoint_step_120.pt` | `34d3df9874da7293effff5d510b264f2b48fb130029ca7a9b46398f81a858c9f` |
| **Exp02 Parent Evaluation Output** | `experiments/research_v08_json_sft/exp02_string_copy/PARENT_BASELINE_RESULTS.json` | `2ae4cff6cbd05600558a0a265e9dcc6187ae238f93eac81c98a6234be6aa1ebd` |
| **Exp02 Candidate Evaluation Output** | `experiments/research_v08_json_sft/exp02_string_copy/CANDIDATE_EVALUATION_RESULTS.json` | `55e566a0ca8f8e9013f4227883abe429153e54551a9f17c4bca2bd78222a311f` |

Protected invariant confirmation:
- Canonical parent checkpoint: **NEVER modified or overwritten**.
- Canonical tokenizer: **NEVER modified or overwritten**.
- `chakrview/runtime/session.py`: **NEVER modified, staged, or touched**.
- Git operations: **Zero commits, branches, merges, or remote operations**.

---

## 3. Dataset Construction & Integrity Audit

The Experiment 02 dataset isolates **single-field string extraction**:
$$\text{Input: Prompt containing status string } S \implies \text{Output: } \{\text{"status": } S\}$$

- **Train Split**: 130 records (100 single-field JSON extraction + 30 continuation replay).
- **Val Split**: 30 records (20 single-field JSON extraction + 10 continuation replay).
- **Test Split**: 30 records (20 single-field JSON extraction + 10 continuation replay).
- **Strict Disjoint Partitioning**:
  - 40 Train Statuses (single and underscored compound words: `nominal`, `active`, `carrier_locked`, `chamber_vacuum`, etc.).
  - 10 Val Statuses (disjoint: `standby_normal`, `frequency_drift`, `voltage_spike`, etc.).
  - 15 Test Statuses (strictly disjoint from train & val: `optical_alignment_fault`, `beam_misaligned`, `cavity_depressurized`, `flux_fluctuation`, `cryo_quench`, etc.).
  - Component entities strictly disjoint across train, val, and test.
- **5 Diverse Instruction Templates**:
  - T1: `"Extract the system status as a JSON object with key \"status\"."`
  - T2: `"Parse the telemetry status from the reading below into a JSON object with the \"status\" field."`
  - T3: `"Convert the following status report into a JSON dictionary containing \"status\"."`
  - T4: `"Given the monitored system state, return a JSON object structured as {\"status\": string}."`
  - T5: `"Identify the reported status condition and output it as JSON: {\"status\": \"<value>\"}."`
- **Validation**: Passed `validate_exp02_dataset.py` with zero errors, zero sequence overflows ($\le 256$ tokens), and zero leakage across splits.

---

## 4. Fresh Parent Baseline (Step-10000)

Evaluated the canonical parent on the frozen Experiment 02 test suite under greedy deterministic decoding ($T=0.0$, `max_new_tokens=64`, stop token `[1]`):

- **JSON Parse Rate**: **0 / 20 (0.0%)**
- **Schema Valid Rate**: **0 / 20 (0.0%)**
- **Exact Copied-String Accuracy**: **0 / 20 (0.0%)**
- **Empty Outputs**: 0 / 20
- **Max-Tokens Truncations**: 20 / 20 (100.0%)
- **Behavior Summary**: Emits repetitive docstring boilerplate (`\n\n... >>> hexited_pathex_reckin(")...`), demonstrating zero initial structured JSON capability.

---

## 5. Training Dynamics (Bounded 120 Steps)

Trained from the canonical parent for 120 steps on CPU (19.04 seconds):
- **Initial Val Loss**: `5.3074`
- **Step 10**: Train Loss `5.7361` | Val Loss `4.8223` | Drift `0.20%`
- **Step 30**: Train Loss `3.4469` | Val Loss `3.8767` | Drift `0.83%`
- **Step 60**: Train Loss `3.0844` | Val Loss `3.7056` | Drift `1.47%`
- **Step 90**: Train Loss `2.8670` | Val Loss `3.4277` | Drift `1.78%`
- **Step 120 (Best Step)**: Train Loss `2.5645` | Val Loss `3.3230` | Drift `1.87%`
- **Safeguards**: 0 NaN/Inf, max grad norm clipped to 1.0, weight drift 1.87% (boundary 4.0%), zero divergence aborts.

---

## 6. Head-to-Head Comparison: Parent Baseline vs Trained Candidate

Evaluated on the exact frozen 30-case held-out test suite ($N=20$ JSON extraction prompts, $N=10$ continuation replay prompts) under identical greedy deterministic decoding:

| Metric Dimension | Canonical Parent Baseline (`Step-10000`) | Trained Candidate (`exp02_best_checkpoint.pt`) | Relative Gain | Result Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **JSON Parse Rate** | **0 / 20 (0.0%)** | **19 / 20 (95.0%)** | **+95.0%** | Massive syntax acquisition. |
| **Exact Schema Rate (`{"status": str}`)** | **0 / 20 (0.0%)** | **19 / 20 (95.0%)** | **+95.0%** | Schema compliance achieved. |
| **Exact Copied-String Accuracy** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** | **+0.0%** | In-context copy failure. |
| **Full Semantic Task Success** | **0 / 20 (0.0%)** | **0 / 20 (0.0%)** | **+0.0%** | No value binding to context. |
| **Empty Outputs** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 | Reliable generation. |
| **Immediate EOS Rate** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 | Sustained sequence length. |
| **Max-Tokens Truncation Rate** | 20 / 20 (100.0%) | 1 / 20 (5.0%) | -95.0% | Candidate emits `}` + `<EOS>` properly. |
| **Replay Non-Empty Rate** | 10 / 10 (100.0%) | 10 / 10 (100.0%) | 0 | Continuity preserved. |
| **Replay Meaningful Continuation** | 2 / 10 (20.0%) | 3 / 10 (30.0%) | +10.0% | Base pretraining capability intact. |

---

## 7. Forensic Analysis of Candidate Generations

### Actual Generated Test Outputs:
```text
exp02_test_json_001 | Expected: optical_alignment_fault | Output: {"status": "path."}
exp02_test_json_002 | Expected: beam_misaligned         | Output: {"status": "pated"}
exp02_test_json_003 | Expected: cavity_depressurized    | Output: {"status": "path."}
exp02_test_json_004 | Expected: flux_fluctuation        | Output: {"status": "pathalso "}
exp02_test_json_005 | Expected: cryo_quench             | Output: {"status": "blicath"}
exp02_test_json_006 | Expected: magnetic_leakage        | Output: {"status": "path."}
exp02_test_json_007 | Expected: harmonic_distortion     | Output: {"status": "pated"}
exp02_test_json_008 | Expected: plasma_disruption       | Output: {"status": "path."}
exp02_test_json_009 | Expected: lattice_strain          | Output: {"status": "path."}
exp02_test_json_010 | Expected: dielectric_breakdown    | Output: {"status": "b"}
exp02_test_json_011 | Expected: core_undervoltage       | Output: {"status": "path."}
exp02_test_json_012 | Expected: resonance_shift         | Output: {"status": "pated"}
exp02_test_json_013 | Expected: thermal_runaway         | Output: {"status": "path."}
exp02_test_json_014 | Expected: quantum_decoherence     | Output: {"status": "path"}
exp02_test_json_015 | Expected: manifold_rupture        | Output: {"status": "non_to_to_to_to_of_to...
exp02_test_json_016 | Expected: optical_alignment_fault | Output: {"status": "path."}
exp02_test_json_017 | Expected: beam_misaligned         | Output: {"status": "pated"}
exp02_test_json_018 | Expected: cavity_depressurized    | Output: {"status": "path."}
exp02_test_json_019 | Expected: flux_fluctuation        | Output: {"status": "pathannot "}
exp02_test_json_020 | Expected: cryo_quench             | Output: {"status": "path"}
```

### Empirical Discovery:
1. **The Single-Field Schema is Solved**:
   - In Exp 01, the model bypassed `"status"` to emit `{"code": 3}`.
   - In Exp 02, when trained exclusively on the single-field schema, the model learned the exact schema: `{"status": "..."}` on **19 out of 20 (95%)** held-out test prompts.
   - It properly opens with `{`, emits `"status": "`, closes the string with `"`, closes the object with `}`, and terminates with `<EOS>`.
2. **The In-Context Copying Failure Mode**:
   - Instead of attending back to the context string (`optical_alignment_fault`, `beam_misaligned`), the model outputs hallucinated/generic fragments (`"path."`, `"pated"`, `"blicath"`).
   - Even on training set examples (e.g. `discharged`, `throttled`), generation outputs `"path."` or `"nd"`.
   - **Root Cause**: ChakrMicro's pure causal self-attention without an explicit pointer-copy mechanism or specialized induction head training cannot route multi-token string spans across $\sim 50$ tokens from prompt context into the generation stream under vanilla next-token cross-entropy within 120 steps. The model minimizes loss by learning the common subword bi-gram priors (`p`, `at`, `ed`, `path`) rather than performing dynamic span retrieval.

---

## 8. Regression Testing

Executed relevant existing project test suites:
- `tests/test_batch107_110_copy_benchmark.py`: **7/7 PASSED**.
- `tests/test_batch171_180_pointer_learning.py`: **5/5 PASSED**.
- Overall regression suite: **12 / 12 passed in 2.00s**, verifying zero regressions in repository components.

---

## 9. Conclusion & Decision Rule Status

In accordance with Section 7 ("Decision rule: If syntax improves but copying remains poor, investigate target supervision, data coverage, optimization, and tokenization before increasing model size or training duration"):

- Syntax and schema improved from **0.0% to 95.0%**.
- In-context string copying remained at **0.0%**.
- The hypothesis that vanilla causal SFT alone can induce general string copying in ChakrMicro without architectural pointer mechanisms or specialized copy objectives is not supported.

### Status:
`PARTIAL IMPROVEMENT — FURTHER DIAGNOSIS REQUIRED`
