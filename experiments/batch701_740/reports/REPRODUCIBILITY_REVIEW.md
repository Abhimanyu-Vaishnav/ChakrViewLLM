# Reproducibility Review: Batches 701–740
## Neural Subtask Decomposition, Generalization & Learnability Investigation

**Review Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `a9eb0be`  

---

### 1. Artifact & Checkpoint Lineage

| Checkpoint Name | File Path | Expected SHA-256 | Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Batch 541–580 Intent Candidate** | `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt` | `9c3204906e8e...` | `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` | **VERIFIED MATCH** |
| **Batch 581–620 Span Pointer** | `experiments/batch581_620/checkpoints/candidate_step250_span_pointer.pt` | `196384e9c29c...` | `196384e9c29ca1042ecf65e379972257bdcbce1667f4befb0e4249e3f7c5b569` | **VERIFIED MATCH** |
| **Batch 701–740 Neural Planner** | `experiments/batch701_740/checkpoints/candidate_step150_neural_planner.pt` | `c4e172c8c182...` | `c4e172c8c182f09cdca60ed925254aca59c558b503ff96bfb279e4e07398fec3` | **VERIFIED CREATED** |

---

### 2. Dataset Provenance & Zero-Leakage Verification

Datasets generated with seed 42 via `experiments/batch701_740/build_decomposition_dataset.py`:
- `train_decomposition.jsonl` (N=70)
- `val_decomposition.jsonl` (N=15)
- `test_decomposition_heldout.jsonl` (N=25)

**Leakage Audit Result:** Exact prompt overlap across train and test sets: **0 (0.0%)**. Disjoint template families confirmed: 3-hop arithmetic, Hindi/Hinglish instructions, and unknown tool requests appear strictly in the test set.

---

### 3. Exact Commands to Reproduce Findings

To reproduce all benchmarks, training runs, and tests from clean Python shell:

```bash
# 1. Generate Decomposition Datasets
python -m experiments.batch701_740.build_decomposition_dataset

# 2. Train Multi-Task Decomposition Model across Seeds
python -m experiments.batch701_740.train_decomposition_candidate

# 3. Evaluate Held-Out Generalization & Causal Interventions
python -m experiments.batch701_740.run_neural_planning_evaluation

# 4. Run Unit and Integration Tests (6 passed)
python -m pytest tests/test_batch701_740_neural_decomposition.py -v

# 5. Run Full Comprehensive Regression Suite across Batches 541 to 740 (54 passed)
python -m pytest tests/test_batch541_580_neural_intent.py tests/test_batch581_620_span_pointer.py tests/test_batch621_660_multiturn_context.py tests/test_batch661_700_dialogue_planning.py tests/test_batch701_740_neural_decomposition.py tests/test_persistent_memory.py -v
```

---

### 4. Hardware Latency & Resource Utilization

- **Host Hardware:** NVIDIA RTX A1000 6GB Laptop GPU (CUDA active).
- **Inference Latency:** **11.71 ms** per planning request (measured over 50 iterations with CUDA synchronization).
- **Memory Footprint:** Peak VRAM < 250 MB.
- **User Working Tree State:** `chakrview/runtime/session.py` and untracked docs/experiment folders preserved without modification.
