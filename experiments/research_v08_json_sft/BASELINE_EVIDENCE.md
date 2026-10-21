# CHAKRVIEW v0.8: CANONICAL PARENT BASELINE EVIDENCE PACKET
## Research Experiment: Single-Task Structural JSON Synthesis & In-Context Entity Binding

- **Execution Date**: 2026-10-21
- **Status**: BASELINE COMPLETE — AWAITING HUMAN TRAINING AUTHORIZATION
- **Authority**: Read-only baseline measurement. Zero training, zero backpropagation, zero optimizer steps, and zero model weight changes executed.

---

## 1. Frozen Dataset Manifest and Cryptographic Hashes

The 200-record dataset was deterministically generated (`random_seed=42`) using `experiments/research_v08_json_sft/generate_dataset.py` and cryptographically locked.

| Split | File Path | Total Records | JSON Tasks | Replay Tasks | Verified SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | `experiments/research_v08_json_sft/data/train.jsonl` | 140 | 100 | 40 | `666bab3401532f757bf73aad2edae9ebc7da85cf258edde135e9afa88a790b0a` |
| **Validation** | `experiments/research_v08_json_sft/data/val.jsonl` | 30 | 20 | 10 | `b7814fac8a65f1b36e8b2a17a322e0d0f071e42b15bfc72813500ad67e25f7f5` |
| **Held-Out Test** | `experiments/research_v08_json_sft/data/test.jsonl` | 30 | 20 | 10 | `b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890` |
| **Manifest** | `experiments/research_v08_json_sft/data/manifest.json` | - | - | - | `91ea2803b306b3bc4f1a26b2df4a5f4bb572f4f1ce2b9f93ee74f2617f1a3b3f` |

---

## 2. Dataset Validation Results & Leakage Audit

Automated validation was executed via `experiments/research_v08_json_sft/validate_dataset.py`. All integrity gates passed with zero violations:

1. **Exact & Normalized Duplicate Checks**:
   - Zero prompt collisions between Train, Val, and Test.
   - Normalized (lowercased, whitespace-collapsed) overlap across splits: **0%**.
2. **Component Entity Leakage**:
   - Train Components (100): `subsystem_alpha_01-25`, `hydraulic_actuator_01-25`, `power_converter_01-25`, `thermal_sensor_01-25`.
   - Val Components (20): `propulsion_thruster_01-10`, `solar_inverter_01-10`.
   - Test Components (20): `cryo_pump_01-10`, `optical_relay_01-10`.
   - Intersection: **0 items (100% disjoint)**.
3. **Status Value Leakage**:
   - Train Statuses (20): `nominal`, `active`, `degraded`, `overheated`, `idle`, `restarting`, `calibrated`, `throttled`, etc.
   - Val Statuses (4): `standby_normal`, `frequency_drift`, `voltage_spike`, `pressure_drop`.
   - Test Statuses (4): `standby_degraded`, `beam_misaligned`, `cavity_depressurized`, `flux_fluctuation`.
   - Intersection: **0 items (100% disjoint)**.
4. **Integer Code Range Partitioning**:
   - Train range: `[106, 696]` (subset of `[100, 699]`).
   - Val range: `[706, 796]` (subset of `[700, 799]`).
   - Test range: `[800, 898]` (subset of `[800, 899]`).
   - Intersection: **0 items (100% disjoint)**.
5. **Replay Corpus Disjointness**:
   - 40 Train facts, 10 Val facts, and 10 Test facts share 0 text overlap and contain 0 telemetry/schema references.
6. **Sequence Length & Truncation Check**:
   - Max observed token length: **134 tokens** (context limit: 256 tokens). Capacity utilization: 52.3%, zero truncation risk.

---

## 3. Evaluator Unit-Test Results

The strict evaluation engine (`experiments/research_v08_json_sft/evaluate_checkpoint.py`, Version `v0.8.0-json-strict`) was verified via `tests/test_v08_json_evaluator.py`:
- Command: `.venv\Scripts\pytest tests/test_v08_json_evaluator.py -v`
- Result: **11 passed in 4.28s, 0 failures, 0 skipped**.
- Assertions verified:
  - Perfect JSON matching target -> Parse Pass, Schema Pass, Status Pass, Code Pass, Full Semantic Pass.
  - Wrong status value -> Parse Pass, Schema Pass, Status Fail, Full Semantic Fail.
  - Wrong code value -> Parse Pass, Schema Pass, Code Fail, Full Semantic Fail.
  - Extra hallucinated keys -> Schema Fail.
  - Missing keys -> Schema Fail.
  - Inverted/wrong primitive types -> Schema Fail.
  - Boolean for code integer (`true` instead of `int`) -> Schema Fail.
  - Syntax corruption (unclosed brackets/quotes) -> Parse Fail.
  - Empty outputs & immediate EOS -> Rejection / Semantic Fail.
  - Markdown fenced JSON -> Handled and stripped cleanly.
  - Replay repetition threshold: `repetition_ratio < 0.40` is required for meaningful continuation.

---

## 4. Canonical Parent Checkpoint & Tokenizer Lineage Verification

Prior to evaluation, all canonical assets were hashed directly from disk:

| Artifact Role | File Path | Verified SHA-256 Checksum | Lineage Match |
| :--- | :--- | :--- | :---: |
| **Canonical Parent Model** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **Bit-Exact Match** |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **Bit-Exact Match** |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **Bit-Exact Match** |
| **Frozen Test Set** | `experiments/research_v08_json_sft/data/test.jsonl` | `b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890` | **Bit-Exact Match** |

---

## 5. Measured Parent Baseline Metrics (Direct Evaluation)

The canonical parent checkpoint was evaluated on the frozen 30-case test set using greedy deterministic decoding ($T=0.0$, `max_new_tokens=64`, stop token `[1]`).

Full results are recorded in:
`experiments/research_v08_json_sft/PARENT_BASELINE_RESULTS.json`.

### Quantitative Metrics with Exact Numerators & Denominators

| Metric Dimension | Canonical Parent Baseline Score | Acceptance Target for Future Candidate | Comparison Interpretation |
| :--- | :---: | :---: | :--- |
| **JSON Parse Rate** | **0 / 20 (0.0%)** | $\ge 10 / 20$ (50.0%) | Parent emits prose, never valid JSON. |
| **Schema Match Rate** | **0 / 20 (0.0%)** | $\ge 8 / 20$ (40.0%) | Parent has zero two-key schema awareness. |
| **Status Value Match** | **0 / 20 (0.0%)** | Evaluated diagnostic | Parent cannot extract or bind status. |
| **Code Value Match** | **0 / 20 (0.0%)** | Evaluated diagnostic | Parent cannot extract or bind integer code. |
| **Full Semantic Task Success** | **0 / 20 (0.0%)** | $\ge 5 / 20$ (25.0%) | Complete baseline floor established. |
| **Empty Outputs** | **0 / 20 (0.0%)** | 0 / 20 | Parent never emits empty string after `Response:`. |
| **Immediate EOS** | **0 / 20 (0.0%)** | $< 2 / 20$ | Instruction formatting avoids immediate EOS. |
| **Max Tokens Truncated** | **20 / 20 (100.0%)** | Evaluated diagnostic | Parent runs into 64-token cap in prose loops. |
| **Replay Non-Empty Rate** | **10 / 10 (100.0%)** | 10 / 10 (100.0%) | Continuation text generated on all replay prompts. |
| **Replay Meaningful Continuation** | **0 / 10 (0.0%)** | $\ge 8 / 10$ (80.0%) | Parent baseline collapses into repetitive loops (mean bigram repetition ratio = **0.605**). |

---

## 6. Representative Output Evidence

### Case 1: JSON Task Failure via Pretraining Python Prose Attractor
- **Prompt ID**: `TEST_JSON_001`
- **Prompt**:
  ```text
  Instruction: Extract the system status and status code as a JSON object with keys "status" and "code".
  Context: Telemetry monitor report: Component cryo_pump_01 reported status 'standby_degraded' with system code 832 during periodic scan.
  Response:
  ```
- **Ground Truth Target**:
  ```json
  {"status": "standby_degraded", "code": 832}
  ```
- **Parent Baseline Generated Text**:
  ```text
  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n    >>> hex_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_of
  ```
- **Scoring**: `json_parsed=False`, `schema_valid=False`, `full_semantic_success=False`.

### Case 2: JSON Task Failure via Document Heading Prose Attractor
- **Prompt ID**: `TEST_JSON_002`
- **Prompt**:
  ```text
  Instruction: Parse the telemetry reading below into JSON format with "status" and "code" fields.
  Context: Telemetry monitor report: Component cryo_pump_02 reported status 'beam_misaligned' with system code 870 during periodic scan.
  Response:
  ```
- **Ground Truth Target**:
  ```json
  {"status": "beam_misaligned", "code": 870}
  ```
- **Parent Baseline Generated Text**:
  ```text
  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n# CChandafts\n# Test is the first of the filevision of the filevision of the fileaderite.\n\n# If 
  ```
- **Scoring**: `json_parsed=False`, `schema_valid=False`, `full_semantic_success=False`.

### Case 3: Replay Degeneration via Repetitive Loop
- **Prompt ID**: `TEST_REPLAY_001`
- **Prompt**:
  ```text
  Empirical Science Series [Cosmology 26]: Hubble's law indicates that galaxies recede from Earth with velocities 
  ```
- **Expected Ground Truth Continuation**:
  ```text
  proportional to their proper cosmological distance.
  ```
- **Parent Baseline Generated Text**:
  ```text
  in the world, which is the world of the world. It is the world of the world, which is the world, which is a show of the word of the wayover of the wor
  ```
- **Scoring**: `replay_non_empty=True`, `repetition_ratio=0.5238` ($\ge 0.40$), `replay_meaningful=False`.

---

## 7. Known Limitations and Scientific Boundary

1. **True Pretraining Floor**: The canonical parent baseline is definitively established as **0/20 (0.0%)** on JSON extraction and **0/10 (0.0%)** on meaningful replay continuation. Any future candidate that scores $>0$ demonstrates genuine structural capability acquisition over the baseline.
2. **Sample Size ($N=20$)**: 20 held-out JSON test prompts provide a rigorous directional signal for pilot-scale structural learning, but cannot support claims of general LLM reasoning.
3. **Template Generalization Boundary**: 4 instruction templates test robustness against minor syntactic phrasing shifts, but do NOT test general open-ended instruction following.

---

## 8. Exact Reproducibility Command

The parent baseline evaluation can be reproduced at any time using:
```powershell
.venv\Scripts\python experiments/research_v08_json_sft/run_parent_baseline.py
```
This script asserts preflight SHA-256 hashes of the checkpoint, tokenizer, and frozen test set before execution, guaranteeing 100% reproducible bit-exact evaluation.

---

## 9. Absolute Safety & Training Invariants Confirmation

- **Zero training executed**: No backpropagation, no optimizer steps, no weight updates.
- **Canonical artifacts untouched**: Parent checkpoint and tokenizer remain 100% unmodified and bit-exact.
- **Protected file untouched**: `chakrview/runtime/session.py` was never opened, modified, staged, or touched.
- **Zero Git mutations**: No commit, push, merge, branch, or tag operations.

---

## FINAL STATUS

`BASELINE COMPLETE — AWAITING HUMAN TRAINING AUTHORIZATION`
