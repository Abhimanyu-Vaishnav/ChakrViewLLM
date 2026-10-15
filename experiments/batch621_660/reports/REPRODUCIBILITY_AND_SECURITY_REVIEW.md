# Reproducibility and Security Review: Batches 621–660
## Multi-Turn Context Binding, Neural Anaphora Resolution & Persistent Memory Grounding

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `9e633a0` (`feat(batch581-620): train span pointer candidate, ground exact arguments, and evaluate 59-item benchmark`)  

---

### 1. Artifact & Checkpoint Cryptographic Lineage

| Checkpoint Name | File Path | Expected SHA-256 | Measured SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Batch 541–580 Intent Candidate** | `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt` | `9c3204906e8e...` | `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` | **VERIFIED MATCH** |
| **Batch 581–620 Span Pointer** | `experiments/batch581_620/checkpoints/candidate_step250_span_pointer.pt` | `196384e9c29c...` | `196384e9c29ca1042ecf65e379972257bdcbce1667f4befb0e4249e3f7c5b569` | **VERIFIED MATCH** |

---

### 2. Dataset Reproducibility & Zero Leakage

All multi-turn datasets were generated via `experiments/batch621_660/build_multiturn_dataset.py` with fixed random seed 42.

- `experiments/batch621_660/data/train_multiturn.jsonl`: 11 scenarios across math, text, chains, and abstentions.
- `experiments/batch621_660/data/val_multiturn.jsonl`: 2 scenarios.
- `experiments/batch621_660/data/test_multiturn_heldout.jsonl`: 16 scenarios across all major scenario families.

**Leakage Audit Result (`experiments/batch621_660/reports/DATASET_LEAKAGE_AUDIT.json`):**
- Exact prompt overlap between train and test: **0 (0.0%)**
- Disjoint templates verified: Test set contains completely held-out multi-hop chains, Hindi/Hinglish phrasing (`SOV` structure), and cross-session isolation queries not present in training.

---

### 3. Evaluation Reproduction Commands

To reproduce the full suite from clean terminal:

```bash
# 1. Multi-Turn Context & Anaphora Benchmark (16 held-out scenarios)
python -m experiments.batch621_660.run_multiturn_comparison

# 2. Capability Demonstration & Adversarial Causal Intervention Suite
python -m experiments.batch621_660.run_adversarial_and_demonstrations

# 3. Batch 621-660 Unit & Integration Tests (7 passed)
python -m pytest tests/test_batch621_660_multiturn_context.py -v

# 4. Multi-Batch Regression Test Suite (41 passed)
python -m pytest tests/test_batch541_580_neural_intent.py tests/test_batch581_620_span_pointer.py tests/test_batch621_660_multiturn_context.py tests/test_persistent_memory.py -v
```

---

### 4. Hardware Adaptation & Performance

- **Environment:** Windows 11, Python 3.14.7, PyTorch 2.6.0+cu124
- **Accelerator:** NVIDIA RTX A1000 6GB Laptop GPU (CUDA available and active)
- **Multi-Turn Context Resolution Latency:** **2.02 ms** per turn average (measured over 50 iterations).
- **Throughput:** ~495 turns/sec.

---

### 5. Security and Privacy Review

1. **Session Boundary Isolation:**
   - Evaluated via `test_session_boundary_isolation` and adversarial test `cross_session_infiltration`.
   - In both cases, queries in Session B attempting to reference Session A's computations ("previous answer", "other user") are blocked. The resolver safely abstains with `Clarification required` and leaks 0 bytes of antecedent data.
2. **Missing Context Abstention vs. Hallucination:**
   - Single-turn baseline hallucinated answers or threw errors on follow-up references when context was missing.
   - `ContextReferenceResolver` achieved 100% abstention on missing-context queries (`missing_context_abstention`), cleanly prompting the user for the missing antecedent.
3. **User Working Tree Preservation:**
   - Uncommitted modifications to `chakrview/runtime/session.py` and untracked scratch files were strictly preserved and not overwritten.
