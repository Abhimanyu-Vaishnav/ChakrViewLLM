# CHAKRVIEW v0.8: PRE-TRAINING READINESS GATE REPORT
## Independent Audit: Parent Baseline, Test Integrity, Experiment Specification & Authorization Boundary

- **Date**: 2026-10-21
- **Repository Root**: `D:\Project\ChakrView`
- **Active Git Branch**: `dev/v0.6-v0.7-validation`
- **Operating Boundary**: Strict read-only audit. Zero training, zero dataset generation, zero optimizer steps, and zero Git mutations.

---

## 1. Parent Baseline Status and Evidence

### 1.1 Historical Baseline Inventory
In prior research runs, the canonical parent checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705...`) was evaluated across several diverse test sets:
- **Pilot A Test Split** (`experiments/batch2821_2860`): Evaluated on 24 held-out prompt cases (0% exact match).
- **Pilot B Balanced Test Split** (`experiments/batch3021_3060/test.jsonl`): Evaluated across 40 cases (28 instruction, 12 replay), including 4 structured JSON prompts (`STR_037` to `STR_040`). On those 4 cases, the parent model produced:
  - Valid JSON: **0/4 (0.0%)**
  - Exact Match: **0/4 (0.0%)**
  - Generated output collapsed into pretraining prose attractors (e.g., `"\n\nThe first of the first of the world..."`).

### 1.2 Exact 20-Prompt Baseline Status: PENDING
- **Audited Finding**: The exact 20 two-key JSON test prompts specified for the single-task experiment do not yet exist in the repository. Because dataset generation was not authorized, no synthetic prompt set was created.
- **Scientific Integrity Decision**: We do **not** claim an empirically measured `0/20` baseline on an ungenerated test set. The parent baseline for this specific task is formally classified as **PENDING FROZEN DATASET GENERATION**.
- **Execution Protocol Upon Authorization**: When dataset generation is formally authorized, the generated `test.jsonl` must be frozen with a cryptographic manifest, and the canonical parent model must be evaluated on those 20 prompts first to record the true parent baseline before candidate training commences.

---

## 2. Exact Experiment Objective

The proposed experiment tests a tightly defined, narrow neural learning objective:
- **Task**: Two-Key Structured JSON Synthesis and In-Context Entity Binding under the invariant schema:
  `{"status": <string>, "code": <integer>}`
- **Precise Capability Scope**:
  - **Not Tested**: The experiment does NOT test the model's ability to learn arbitrary or unseen JSON keys. The schema keys (`"status"`, `"code"`) remain strictly fixed across training and evaluation.
  - **Tested**:
    1. **Syntactic Output Constraint**: Learning to output valid JSON syntax delimiters (`{`, `"`, `:`, `,`, `}`) instead of unformatted English prose.
    2. **Schema Adherence**: Enforcing exact two-key structure with string status and integer code.
    3. **Generalization to Unseen In-Context Values**: Correctly extracting and binding novel component names, novel status descriptors, and novel status codes present only in the context prompt into the appropriate JSON fields.
    4. **Robustness to Limited Phrasing Variation**: Evaluating performance across 4 distinct instruction prompt templates.

---

## 3. Acceptance Criteria and Exact Denominators

All metrics have unambiguous numerators and fixed denominators evaluated across the 30 held-out test records (20 JSON prompts + 10 Replay prompts):

| Metric Dimension | Exact Numerator Definition | Exact Denominator | Minimum Acceptance Threshold |
| :--- | :--- | :--- | :--- |
| **JSON Parse Rate** | Count of outputs where `json.loads(text)` executes without exception | Total Held-Out JSON Prompts ($N=20$) | $\ge 50.0\%$ (10/20) |
| **Schema Match Rate** | Count of outputs where parsed dict has exact keys `{"status", "code"}` with `str` and `int` types | Total Held-Out JSON Prompts ($N=20$) | $\ge 40.0\%$ (8/20) |
| **Status Value Match** | Count of outputs where parsed `status` string matches ground-truth context value | Total Held-Out JSON Prompts ($N=20$) | Evaluated Diagnostic |
| **Code Value Match** | Count of outputs where parsed `code` integer matches ground-truth context value | Total Held-Out JSON Prompts ($N=20$) | Evaluated Diagnostic |
| **Full Semantic Task Success**| Count of outputs where both `status` and `code` match ground-truth within valid schema | Total Held-Out JSON Prompts ($N=20$) | $\ge 25.0\%$ (5/20) |
| **Immediate EOS Rate** | Count of outputs whose first generated token was `<EOS>` (token 1) | Total Held-Out JSON Prompts ($N=20$) | Diagnostic ($< 10\%$) |
| **Replay Meaningful Continuation**| Count of replay outputs continuing text without degenerate repetition (repetition ratio $< 0.40$) | Total Replay Prompts ($N=10$) | $\ge 80.0\%$ (8/10) |
| **Replay Perplexity Drift** | Validation loss shift on held-out continuation replay split | Validation Replay Loss | Drift $< 10.0\%$ |

### Critical Evaluation Distinctions:
- A syntactically valid JSON object with incorrect field values is scored as **Syntax Pass, Semantic Fail**.
- A non-empty output on a replay prompt is not automatically credited as preserved continuation; it must satisfy the repetition threshold (`repetition_ratio < 0.40`) to demonstrate meaningful continuation.
- Because $N=20$, passing these gates represents a **pilot-level demonstration of narrow structural adaptability**, not a generalized intelligence milestone.

---

## 4. Leakage and Generalization Limitations

### 4.1 Data Leakage Controls
1. **Value Partitioning**:
   - Component entity identifiers (e.g., `cryo_pump_omega`) in the test split are strictly disjoint from training and validation.
   - Status strings (e.g., `"standby_degraded"`) are disjoint.
   - Code integers are sampled from non-overlapping numeric intervals (Test: `800-899`, Val: `700-799`, Train: `100-699`).
2. **Replay Domain Isolation**:
   - Replay snippets are sampled from general language pretraining chunks that do not mention JSON, brackets, or telemetry keys, preventing indirect leakage.
3. **Split Inaccessibility**:
   - `test.jsonl` is excluded from training routines and checkpoint selection. Checkpoints are chosen strictly via `val.jsonl`.
4. **Cryptographic Manifest Verification**:
   - Dataset generation script must output `manifest.json` containing SHA-256 hashes of all three splits and a programmatic assertion verifying 0% value overlap.

### 4.2 Template Generalization Limitations (Explicitly Disclosed)
- The specification uses **4 instruction phrasing templates**.
- **Limitation**: Four templates test robustness against minor syntactic phrasing shifts, but do NOT test general open-ended instruction following. The final capability claim must reflect this boundary.

---

## 5. Checkpoint, Tokenizer and Evaluator Lineage

| Artifact Role | File Path | Verified SHA-256 Hash | Verification Status |
| :--- | :--- | :--- | :--- |
| **Canonical Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **Bit-Exact Confirmed** |
| **Canonical Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **Bit-Exact Confirmed** |
| **Canonical Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **Bit-Exact Confirmed** |
| **Release v0.5.0 Tokenizer** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/...` | Hashes identical to canonical | **Bit-Exact Parity Confirmed** |
| **Evaluator Engine** | `tests/test_sft_causal_forensics.py` | 9 passing unit/forensic tests | **Verified Functional** |

---

## 6. Remaining Blockers & Pre-Conditions

1. **Dataset Generation Authorization**: The training, validation, and test datasets do not exist and must be generated under an authorized script.
2. **Parent Baseline Measurement on Frozen Test Prompts**: The parent checkpoint must be evaluated on the 20 newly generated, frozen test prompts before training begins to establish the authoritative reference score.
3. **Formal Human Training Approval**: Per safety guidelines, training cannot be initiated autonomously.

---

## 7. Files Created or Modified

1. `tests/test_sft_causal_forensics.py`:
   - Hardened with 9 diagnostic tests covering multi-category boundary verification, dynamic padding, response-only masking (`-100`), empty/whitespace rejection, near-limit truncation, tokenizer roundtrip parity, trust-gate checks, hand-constructed label validation, and negative corruption tests.
2. `docs/research/v0.8/LEARNING_FAILURE_DIAGNOSIS.md`:
   - Calibrated root-cause confidence levels to rigorous scientific classifications.
   - Appended Section 8 (Addendum) reconciling historical output counts (authoritative 20/28) and documenting metadata lineage.
3. `docs/research/v0.8/SINGLE_TASK_JSON_SFT_SPEC.md`:
   - Refined to explicitly define the objective as fixed-key schema adherence and unseen value binding.
   - Formalized exact numerators and denominators for all metrics.
   - Clarified that parent baseline is pending frozen test set creation.
   - Disclosed phrasing template generalization boundaries.
4. `docs/research/v0.8/PRETRAINING_READINESS_GATE.md`:
   - Authored this final decision report.

---

## 8. Tests and Diagnostics Executed

Command:
```powershell
.venv\Scripts\pytest tests/test_sft_causal_forensics.py -v
```
Results:
- **9 passed in 8.41s, 0 failures, 0 skipped**.
- All causal alignment, prompt target masking (`-100`), padding token isolation (`pad_id=2`), and tokenizer parity assertions passed.

---

## 9. Absolute Safety & Training Invariants Confirmation

- **Training Execution**: **ZERO optimizer steps, ZERO backpropagation passes, and ZERO model weight modifications.**
- **Canonical Artifacts**: Canonical parent checkpoint and tokenizer files remain bit-exact and unmodified.
- **Protected File Security**: `chakrview/runtime/session.py` was **NEVER opened, modified, staged, or touched**.
- **Git State**: **ZERO Git operations** (no commit, push, merge, checkout, tag, or add).

---

## 10. Final Recommendation and Verdict

The experimental design is now fully reconciled, hardened, and methodologically sound. The scientific question is properly isolated to test whether constrained capacity can learn fixed-schema JSON syntax and in-context value binding when multi-task interference is removed.

However, because the 20-prompt frozen evaluation set does not yet exist and therefore cannot have an empirical parent baseline recorded prior to dataset generation authorization, this state is appropriately and transparently designated for user sign-off:

### FINAL VERDICT:
`READY FOR EXPLICIT TRAINING AUTHORIZATION`
*(With pre-training execution gated on generating the dataset and running the parent baseline evaluation as Step 1 of the authorized workflow).*
