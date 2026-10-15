# Reproducibility Review: Batches 741–780
## Autoregressive Plan DSL, Variable-Length Neural Planning & Safe Execution

**Review Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `065061d`  

---

### 1. Checkpoint & Artifact Hashes

| Checkpoint Identifier | File Path | Expected SHA-256 | Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Batch 541–580 Intent Candidate** | `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt` | `9c3204906e8e...` | `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` | **VERIFIED MATCH** |
| **Batch 581–620 Span Pointer** | `experiments/batch581_620/checkpoints/candidate_step250_span_pointer.pt` | `196384e9c29c...` | `196384e9c29ca1042ecf65e379972257bdcbce1667f4befb0e4249e3f7c5b569` | **VERIFIED MATCH** |
| **Batch 701–740 Neural Planner** | `experiments/batch701_740/checkpoints/candidate_step150_neural_planner.pt` | `c4e172c8c182...` | `c4e172c8c182f09cdca60ed925254aca59c558b503ff96bfb279e4e07398fec3` | **VERIFIED MATCH** |
| **Batch 741–780 Autoregressive Candidate** | `experiments/batch741_780/checkpoints/candidate_step250_autoregressive_planner.pt` | `38db3d09668b...` | `38db3d09668bdb980fc72c76da17398d2267644ca698a58a1e6f10d1d392628c` | **VERIFIED CREATED** |

---

### 2. Dataset Lineage & Leakage Audit

Generated via `experiments/batch741_780/build_plan_dsl_dataset.py`:
- `train_plan_dsl.jsonl` (N=100)
- `val_plan_dsl.jsonl` (N=20)
- `test_plan_dsl_heldout.jsonl` (N=30)
- **Prompt Overlap:** 0 (0.0% leakage).

---

### 3. Execution Commands to Reproduce Findings

```bash
# 1. Build Plan DSL Datasets
python -m experiments.batch741_780.build_plan_dsl_dataset

# 2. Train Autoregressive Plan DSL Generator
python -m experiments.batch741_780.train_autoregressive_planner

# 3. Evaluate Generalization, Causal Interventions & Runtime Execution
python -m experiments.batch741_780.run_dsl_generalization_and_runtime

# 4. Run Unit and Integration Tests (6 passed)
python -m pytest tests/test_batch741_780_plan_dsl.py -v

# 5. Run Full Combined Regression Suite across Batches 541 to 780 (60 passed)
python -m pytest tests/test_batch541_580_neural_intent.py tests/test_batch581_620_span_pointer.py tests/test_batch621_660_multiturn_context.py tests/test_batch661_700_dialogue_planning.py tests/test_batch701_740_neural_decomposition.py tests/test_batch741_780_plan_dsl.py tests/test_persistent_memory.py -v
```

---

### 4. Latency & Resource Consumption

- **Hardware:** NVIDIA RTX A1000 6GB Laptop GPU (CUDA active).
- **Autoregressive Generation Latency:** ~517.17 ms per 60 generated tokens (~116 tokens/sec).
- **VRAM Footprint:** < 300 MB.
- **Working Tree:** `chakrview/runtime/session.py` strictly preserved without modification.
