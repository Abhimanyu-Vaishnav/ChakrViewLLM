# CHAKRVIEW BATCH 28 — STAGE D GO/NO-GO AUDIT REPORT

**Audit Date:** 2026-10-11  
**Branch:** `dev/v0.6-v0.7-validation`  
**Working Tree Status:** Preserved without alteration; zero commits or pushes executed.  
**Auditor Role:** Senior ML Systems Architect & Reproducibility Lead  
**Audit Purpose:** Comprehensive evidence-gated verification of the Step 10,000 checkpoint foundation, benchmark provenance, sampler mechanics, supervised loss masking, and instruction data pipeline before committing compute to Stage D instruction tuning.

---

## EXECUTIVE DECISION: `CONDITIONAL GO`

```
+----------------------------------------------------------------------------------------------------+
|                                    BATCH 28 AUDIT GATE SUMMARY                                     |
+--------------------------+----------+--------------------------------------------------------------+
| Audit Gate               | Status   | Core Finding / Verified Metric                               |
+--------------------------+----------+--------------------------------------------------------------+
| 1. Repository State      | PASS     | Clean working tree; zero git resets, commits, or pushes.     |
| 2. Step 10,000 Integrity | PASS     | SHA-256 match bit-exact; 57 tensors loaded; Delta W = 0.     |
| 3. Benchmark Provenance  | PASS     | 12 probes verified against source files; 0 train leakage.    |
| 4. Sampler Semantics     | PASS     | Shard-cluster balancing; Indic draw rate = 50.0% verified.   |
| 5. SFT Loss Masking      | PARTIAL  | Gradient isolation verified (0.0 grad); loop NOT INTEGRATED. |
| 6. SFT Data Pipeline     | PARTIAL  | Schema validator verified; dataset is ILLUSTRATIVE ONLY.     |
| 7. Bounded Smoke Tests   | PASS     | 15/15 tests passing; git diff --check clean.                 |
+--------------------------+----------+--------------------------------------------------------------+
```

**Verdict Rationale**:
The mathematical and scientific foundations for Stage D are sound:
1. The **Step 10,000 checkpoint is authentic and integral**, with bit-exact preservation of canonical baseline initialization ($\Delta W = 0$).
2. The **12-probe linguistic benchmark has strict provenance** and zero 16-gram leakage against all 27 Stage C training shards.
3. The **supervised loss masking logic is mathematically proven at the gradient level** (prompt token gradients are identically 0.0, while response tokens receive gradient updates).

However, an actual SFT training run must **NOT** be launched immediately because:
- The supervised prompt-masked loss function exists in the Stage D preparation module, but has **not yet been integrated into a dedicated SFT training engine/loop**.
- The instruction dataset (`sample_sft_dataset.jsonl`) consists of **4 illustrative samples**, not a vetted training corpus.

---

## 1. GATE 1 — REPOSITORY PRESERVATION
- **Working Branch**: `dev/v0.6-v0.7-validation`.
- **Pre-sprint State**: Verified clean working tree post-commit `394f8b6`.
- **Preservation Policy**: Zero git modifications, resets, stashes, commits, or branch modifications were executed during this audit.

---

## 2. GATE 2 — STEP 10,000 CHECKPOINT INTEGRITY
- **Checkpoint Location**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Computed SHA-256 Digest**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Recorded SHA-256 Digest**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**MATCH**)
- **File Size**: 42,212,787 bytes
- **Model Parameters**: 57 tensors loaded strictly into `ChakrMicro(ModelConfig(max_seq_len=512))`. Zero missing keys, zero unexpected keys.
- **Optimizer State**: AdamW tracking 56 layer states; learning rate `2.0000019e-05`.
- **Canonical Baseline Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` verified bit-exact ($\Delta W = 0$, 3,443,136 parameters).

---

## 3. GATE 3 — BENCHMARK PROVENANCE & LEAKAGE AUDIT
- **Manifest**: `experiments/batch22_27_validation/benchmarks/benchmark_manifest_v1.json` (SHA-256: `9f35196569489403487c6031853db20577aa2f89f41753bb9671607a90f39e31`).
- **Provenance Resolution**:
  - All 12 examples map to verified on-disk files under `data/validation/` (`english.txt`, `hindi.txt`, `hinglish.txt`, `code.txt`, `mathematics.txt`).
- **Contamination & Leakage Audit**:
  - Full sliding 16-token window scan executed across all 27 Stage C training shards (`data/tokenized/stage_c/train/*.bin`).
  - **Result**: Zero 16-gram matches found across all training shards for all 12 probes.
- **Classification**: **HELD-OUT VALIDATION BENCHMARK** (Strict independence established).

---

## 4. GATE 4 — SAMPLER BEHAVIOR & SHARD CLUSTERING
- **Operational Level**: **Shard-Cluster Balancing** (not fine-grained per-document language balancing).
  - Indic Cluster: Shards 9–19 (~2.25M to ~5.0M cumulative token offsets).
  - Non-Indic Cluster: Shards 0–8 and 20–26 (predominantly English crawl and code snippets).
- **Draw Allocation**: In `sampling_mode="balanced"`, draws alternate 50% from the Indic shard offset range and 50% from the non-Indic offset ranges.
- **Boundary Handling**: `MultiShardDataset.get_slice()` cleanly supports boundary straddling via contiguous `np.memmap` buffer concatenation, ensuring no out-of-bounds reads or slice truncations occur.

---

## 5. GATE 5 — SUPERVISED LOSS MASKING (STAGE D)
- **Mathematical Specification**: Autoregressive sequence `<BOS>` + `prompt_tokens` + `response_tokens` + `<EOS>`. Target positions for prompt tokens are masked with `-100`.
- **Gradient-Level Verification**:
  - Executed gradient propagation test with PyTorch `nn.CrossEntropyLoss(ignore_index=-100)`.
  - **Prompt Token Gradients**: Maximum absolute gradient on prompt positions = **`0.00000000`** (exactly zero).
  - **Response Token Gradients**: Maximum absolute gradient on response positions = **`0.33329847`** (active gradient updates).
- **Implementation Status**: **`PARTIAL`**. The collator and loss math are verified in `experiments/batch22_27_validation/stage_d_prep/validator.py`, but a dedicated `SFTExperimentEngine` training loop has not yet been implemented in the training engine.

---

## 6. GATE 6 — SFT DATA PIPELINE
- **Validator Rules**: Enforces non-empty strings, character length minimums, max token context bounds ($\le 512$), valid language codes, and ID uniqueness.
- **Current Dataset Status**: **`PARTIAL (ILLUSTRATIVE ONLY)`**.
  - `sample_sft_dataset.jsonl` contains 4 mock examples for validation testing.
  - A production SFT dataset has not yet been assembled.
  - No synthetic or fabricated data has been labeled as real or trained upon.

---

## 7. GATE 7 — BOUNDED SMOKE TESTS
- **Executed Suites**:
  - `tests/test_pretraining_engine.py`: 6 passed
  - `tests/test_batch22_27_validation.py`: 5 passed
  - `tests/test_batch28_stage_d_gate.py`: 4 passed
- **Total Test Results**: **15 / 15 PASSED (100% pass rate in 7.01s)**.
- **Git Check**: `git diff --check` reported 0 whitespace or formatting errors.

---

## MANDATORY PREREQUISITES BEFORE LAUNCHING SFT PILOT
To transition from `CONDITIONAL GO` to a full `GO` for Stage D training execution, the following two steps must be completed:
1. **Engine Integration**: Implement an `SFTExperimentEngine` in `chakrview/training/` that utilizes `build_masked_sft_tensor` to stream instruction batches with `-100` prompt masking and validation loss tracking.
2. **Dataset Compilation**: Curate a dedicated instruction dataset of at least 200–500 vetted prompt-response pairs across English, Hindi, and Code, passing all validation gates in `StageDDataValidator`.
