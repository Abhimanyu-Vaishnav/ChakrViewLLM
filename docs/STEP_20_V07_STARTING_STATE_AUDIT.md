# CHAKRVIEW v0.7 — STARTING STATE AUDIT (STEP 20)

**Date**: 2026-10-11  
**Repository Root**: `D:\Project\ChakrView`  
**Current Git State**:  
- Branch: `main`
- Latest Commit: `c91302d` (`feat(pretraining): implement v0.6 neural pretraining engine, benchmarks, and persistent learning suite`)
- Working tree cleanliness: Clean (`git status --short` returned 0 modified files).

---

## 1. Verification of Prior v0.6 Implementation and Artifacts

| Component / Artifact | Path | Status | Verification Detail |
| :--- | :--- | :---: | :--- |
| **V0.6 Report** | `docs/STEP_19_V06_NEURAL_PRETRAINING_REPORT.md` | **EXISTS** | Generated and committed in `c91302d` |
| **Pretraining Engine** | `chakrview/training/pretraining_engine.py` | **EXISTS** | Resumable engine, gradient clipping, safety rollbacks |
| **Corpus Audit Script** | `scripts/audit_corpus_quality.py` | **EXISTS** | Milestone 1 audit script |
| **Corpus Audit Data** | `artifacts/corpus_audit_report.json` | **EXISTS** | 32 shards, 7.7M tokens verified |
| **Language Experiments** | `artifacts/milestone4_language_experiments.json` | **EXISTS** | Budgets (0, 30, 60, 100), multi-seed (42, 101, 202) |
| **Generalization Data** | `artifacts/milestone5_generalization_benchmark.json` | **EXISTS** | Inductive suite + ablation & shuffled controls |
| **Cognitive Integration**| `artifacts/milestone6_cognitive_integration_report.json` | **EXISTS** | 100% decision accuracy, fail-closed abstention |
| **Governed Persistence**| `artifacts/milestone7_governed_learning_report.json` | **EXISTS** | Provenance gate, rollback, baseline untouched |
| **Hardware Benchmark** | `artifacts/milestone8_performance_benchmarks.json` | **EXISTS** | CPU (740 tok/s), CUDA (4,735 tok/s) |
| **Objective Tests** | `tests/test_neural_training_objective.py` | **EXISTS** | Causal mask, next-token alignment |
| **Pretraining Tests** | `tests/test_pretraining_engine.py` | **EXISTS** | Resumption, determinism tests |

---

## 2. Canonical Model Architecture & Baseline Verification

Direct execution of `verify_canonical_baseline(ChakrMicro(ModelConfig()))`:
- **Parameter Count**: **3,443,136** (matches canonical expectation exactly)
- **Baseline Weight SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Weight Bit-Exactness**: `True`
- **Baseline Drift**: $\Delta W = 0$

### Exact Architecture Invariants (`chakrview/brain/config.py`):
- `vocab_size`: **4096**
- `d_model`: **192**
- `n_layers`: **6**
- `n_heads`: **6**
- `head_dim`: **32** ($192 / 6 = 32$)
- `hidden_dim`: **512** (SwiGLU intermediate dimension, exactly $(8/3) \times 192$)
- `max_seq_len`: **512**
- `rope_theta`: **10000.0**
- `rms_norm_eps`: **1e-5**
- `tie_embeddings`: **True** (Input embedding matrix tied to LMHead projection)
- `use_bias`: **False** (bias-free linear projections)

### Discrepancy Investigation: "6 heads vs 4 heads"
- **Finding**: In `docs/STEP_19_V06_NEURAL_PRETRAINING_REPORT.md` (and the user prompt prompt text), the high-level summary paragraph inadvertently referenced "4 heads".
- **Code Reality**: The canonical code `ModelConfig.n_heads` is frozen at **6** with `head_dim=32` ($192/6 = 32$). The actual instantiated models in all code (`ChakrMicro`, `verify_canonical_baseline`) use `n_heads = 6`. Attempting to instantiate $n\_heads=4$ with $d\_model=192$ yields $head\_dim=48$, which deviates from the bit-exact 3,443,136 parameter baseline hash `c5571c...00a282da`.
- **Resolution**: In accordance with Rule 2 and Milestone 1 instructions, the canonical 6-head architecture is **strictly preserved**. The v0.6 report text contained a typographical discrepancy in its summary prose, while the underlying code was and remains the canonical 6-head configuration.

---

## 3. Checkpoint Tracking & Governance Invariants

- **Canonical Baseline**: Deterministically instantiated via seed 42 with frozen hash `c5571c...00a282da`. Not stored as a tracked checkpoint file, but dynamically validated in memory.
- **Candidate Checkpoints**: Stored under `artifacts/milestone4_experiments/` (`training_checkpoint_step_30.pt`, etc.). These are ignored by `.gitignore` (`artifacts/*`, `*.pt`), ensuring repository cleanliness.
- **QA Review Queue**: `checkpoints/experiments/neural_learning_v02/stage_d_design/qa_review_queue_v0.3.0.jsonl`
  - Total records: **30**
  - Unreviewed records: **30 (100%)**
  - SHA-256: `c86da067de45d4b2fca966bb5eff85025a2687503acc4acfd98e05307e6acd70`
