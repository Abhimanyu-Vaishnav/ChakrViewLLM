# Starting Point Audit: Batches 821–860
## Neural Pointer-Generator, Exact Argument Grounding & Efficient Grammar-Constrained Decoding

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `c0d13bc` (`feat(batch781-820): implement grammar-constrained autoregressive plan decoding, exact argument audit, and multi-seed evaluation`)

---

### 1. Lineage, Working Tree & Checkpoint Integrity

1. **Canonical Step-10,000 Checkpoint:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Status: **VERIFIED IMMUTABLE AND INTACT**.
2. **Prior Milestone Checkpoints (Batch 781–820):**
   - Seed 42: `experiments/batch781_820/checkpoints/candidate_step300_seed42_grammar_planner.pt` (SHA-256: `e3f2fbb2d2e0aa82992daeb48e821953d69d9404cfd13f0d95e0ddd9478ed4e8`)
   - Seed 101: `experiments/batch781_820/checkpoints/candidate_step300_seed101_grammar_planner.pt` (SHA-256: `74cdb1403673c9680e1ae5c261ed218726c1675fa3ffb495e47498e159408837`)
   - Status: **VERIFIED PRESENT AND INTACT**.
3. **Frozen Benchmark Hash:**
   - Path: `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
   - Measured SHA-256: `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a`
   - Status: **VERIFIED IMMUTABLE**.
4. **User Modifications:**
   - `chakrview/runtime/session.py` remains modified by the user, uncommitted, and untouched.

---

### 2. Independent Reproduction of Prior Milestone Results

- **Held-Out Generalization on Frozen Benchmark ($N=70$):**
  - System A (Fixed-Head Baseline): 37/70 (52.9% - only succeeds on $\le 2$ steps)
  - System B (Unconstrained AR): 37/70 (52.9% syntax validity, 8.6% exact plan match)
  - System C (Grammar-Constrained AR): 42/70 (60.0% syntax validity, 8.6% exact plan match)
- **Argument Copy Fidelity:**
  - Evaluated on $N=50$ validation items: **0.0% (0/42)**.
- **Inference Latency Profile (RTX A1000 GPU):**
  - Unconstrained: p50 = 664.18 ms (180.7 tok/s)
  - Constrained: p50 = 2329.00 ms (51.5 tok/s, ~3.5x overhead)
- **Regression Suite:**
  - 40/40 tests passing across Batches 541–820 in 4.34 seconds.

---

### 3. Mission Objectives for Batches 821–860

1. **Design and Train a Neural Pointer-Generator (`ChakrPointerMicro`):**
   Implement a dual-distribution output head computing $P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) P_{\text{ptr}}(w)$, enabling direct attention-based copying of prompt tokens for argument slots.
2. **Optimize Grammar-Constrained Decoding:**
   Eliminate the 3.5x latency overhead by profiling bottlenecks and implementing an incremental state-transition lookup mechanism.
3. **Combined Evaluation & Provenance Verification:**
   Evaluate the combined pointer-generator + grammar decoder on the $N=70$ frozen benchmark, test causal grounding under operand mutations, and establish an evidence-based promotion decision.
