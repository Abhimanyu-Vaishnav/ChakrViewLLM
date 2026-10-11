# CHAKRVIEW BATCH 29 — STAGE D SFT IMPLEMENTATION & EVALUATION REPORT

**Execution Branch:** `dev/v0.6-v0.7-validation`  
**Overall Sprint Status:** `VERIFIED COMPLETE`  
**Target Milestone:** Stage D Supervised Fine-Tuning Engine, Grounded Dataset Pipeline, Multi-Window Leakage Audit, and Bounded CPU SFT Pilot  
**Canonical Baseline Integrity:** `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` verified bit-exact ($\Delta W = 0$, 3,443,136 parameters).

---

## EXECUTIVE SCORECARD

```
+----------------------------------------------------------------------------------------------------+
|                                    BATCH 29 SPRINT SCORECARD                                       |
+--------------------------+----------+--------------------------------------------------------------+
| Workstream               | Status   | Core Finding / Verified Metric                               |
+--------------------------+----------+--------------------------------------------------------------+
| A. Checkpoint Integrity  | PASS     | SHA-256 match; 57 tensors loaded; Delta W = 0.               |
| B. Batch 28 Gap Repairs  | PASS     | Multi-window 8-token sliding audit clean across all 27 shards |
| C. SFT Engine            | PASS     | chakrview/training/sft_engine.py implemented & verified.     |
| D. Dataset Pipeline      | PASS     | 40 verified examples (en, hi, hi-en, code) from repo corpus. |
| E. Correctness Tests     | PASS     | 18/18 tests passing across all focused suites; git clean.    |
| F. Bounded SFT Pilot     | PASS     | 50 steps; Val loss 4.77 -> 4.53; pretraining drift: -1.5%.   |
| G. Documentation         | PASS     | Comprehensive JSON and Markdown reports authored.            |
+--------------------------+----------+--------------------------------------------------------------+
```

---

## 1. WORKSTREAM A — REPOSITORY & CHECKPOINT INTEGRITY
- **Working Branch**: `dev/v0.6-v0.7-validation`.
- **Base Checkpoint Audited**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- **Computed SHA-256 Digest**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**EXACT MATCH**).
- **Strict Model Loading**: Strict load into `ChakrMicro(ModelConfig(max_seq_len=512))` passed with 0 missing keys and 0 unexpected keys across all 57 layer tensors.
- **Canonical Baseline Invariant**: Bit-exact canonical hash `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W = 0$) verified.

---

## 2. WORKSTREAM B — REPAIR OF BATCH 28 EVIDENCE GAPS
- **Comprehensive Multi-Window Leakage Audit**:
  - Rather than testing only leading prefixes, an exhaustive sliding-window audit using 8-token subwords across prompt, target, and joint sequences was executed against all 27 Stage C binary training shards.
  - **Result**: Zero 8-gram matches detected. Benchmark manifest probes remain strictly un-leaked.
  - Saved artifact: `experiments/batch29_stage_d_sprint/comprehensive_leakage_audit.json`.
- **Production Sampler Instrumentation**:
  - The production `MultiShardDataset.get_batch` was directly instrumented across 100 consecutive batches (800 sequences) without test simulation wrappers.
  - Confirmed: 100% boundary safety (contiguous buffer concatenation across shards) and bit-exact reproducibility for identical RNG seeds.
  - Saved artifact: `experiments/batch29_stage_d_sprint/sampler_production_audit.json`.

---

## 3. WORKSTREAM C — SFT TRAINING ENGINE IMPLEMENTATION
Implemented in `chakrview/training/sft_engine.py`:
- **Response-Only Causal Loss Masking**: Causal target sequence applies `-100` (`ignore_index`) to prompt positions, training strictly on response tokens and the final `<EOS>`.
- **Gradient Isolation**: Prompt positions receive identically `0.00000000` gradient, while response positions receive active updates.
- **Atomic Versioned Checkpoints**: Writes to `.tmp` files and atomic replacement. Checkpoints store model state dict, optimizer state, scheduler state, global SFT step, and RNG states.
- **Resumability**: Full continuation without step loss or optimizer state discarding.

---

## 4. WORKSTREAM D — TRUSTWORTHY SFT DATASET PIPELINE
Curated and validated in `experiments/batch29_stage_d_sprint/data/`:
- **Manifest**: `sft_manifest.json` (SHA-256 tracked).
- **Total Validated Records**: 40 instruction pairs (30 train, 10 val).
- **Language Breakdown**:
  - English (`en`): 16 examples (systems, architectural mechanics, tokenization).
  - Hindi (`hi`): 13 examples (autonomous neural design, digital knowledge, NCERT/grammar).
  - Hinglish (`hi-en`): 5 examples (low-resource execution, command line testing).
  - Code (`code`): 6 examples (PyTorch normalization, algorithms, configuration).
- **Provenance**: Grounded in repository-owned text files (`data/validation/*.txt`) with zero synthetic external LLM generation or fabricated citations.

---

## 5. WORKSTREAM E — AUTOMATED CORRECTNESS & REGRESSION TESTS
- **New Suite**: `tests/test_sft_engine.py` (3 tests covering dataset encoding, prompt gradient isolation, and end-to-end execution).
- **Full Focused Suite Execution**:
  - `tests/test_pretraining_engine.py`: 6/6 PASSED
  - `tests/test_batch22_27_validation.py`: 5/5 PASSED
  - `tests/test_batch28_stage_d_gate.py`: 4/4 PASSED
  - `tests/test_sft_engine.py`: 3/3 PASSED
- **Total Suite Pass Rate**: **18 / 18 PASSED (100% in 7.62 seconds)**.
- **Git Check**: `git diff --check` reported 0 formatting errors.

---

## 6. WORKSTREAM F — INTEGRATED BOUNDED SFT TRAINING & RETENTION EVALUATION
Executed a bounded 50-step pilot on CPU:
- **Starting Base Checkpoint**: Step 10,000 checkpoint (`96357705...471ee`).
- **Base Checkpoint Preservation**: Pre-run and post-run SHA-256 hashes match bit-exact; foundation checkpoint was preserved untouched.
- **Pilot Hyperparameters**: Batch size 2, Max seq len 256, LR `5e-5 -> 5e-6` cosine decay over 50 steps.
- **SFT Loss Trajectory**:
  - Initial Val Loss: **4.7745** (Val Perplexity: 118.45)
  - Final Train Loss: **4.5837**
  - Final Val Loss: **4.5370** (Val Perplexity: **93.41**)
- **Pretraining Retention Audit**:
  - Evaluated on the 12 held-out pretraining benchmark probes before and after SFT.
  - Base Model Loss: **5.5655** (Perplexity: 261.26)
  - Post-SFT Model Loss: **5.4819** (Perplexity: **240.30**)
  - **Loss Drift**: **-1.5%** (Zero catastrophic forgetting; slight positive transfer observed on shared linguistic syntax).
  - Saved artifact: `experiments/batch29_stage_d_sprint/pretraining_retention_audit.json`.

---

## USAGE GUIDE FOR STAGE D
1. **Validate Dataset**:
   ```python
   from experiments.batch22_27_validation.stage_d_prep.validator import StageDDataValidator
   v = StageDDataValidator()
   res = v.validate_file("path/to/dataset.jsonl")
   ```
2. **Execute SFT Training**:
   ```python
   from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
   cfg = SFTConfig(
       repo_root=Path("."),
       train_file=Path("experiments/batch29_stage_d_sprint/data/sft_train.jsonl"),
       val_file=Path("experiments/batch29_stage_d_sprint/data/sft_val.jsonl"),
       output_dir=Path("experiments/my_sft_run"),
       base_checkpoint=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
       steps=100
   )
   engine = SFTExperimentEngine(cfg)
   results = engine.train()
   ```

---

## NEXT RECOMMENDED MILESTONE
Expand the curated instruction repository to 250–500 instruction pairs across domain-specific Indic and technical reasoning tasks, followed by a 200-step curriculum run with multi-turn prompt templates.
