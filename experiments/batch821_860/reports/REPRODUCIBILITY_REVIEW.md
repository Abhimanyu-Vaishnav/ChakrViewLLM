# Reproducibility Review: Batches 821–860
## Commands, Checkpoint Lineage, Cryptographic Hashes & Audit

**Date:** 2026-10-15  
**Milestone:** Batches 821–860  
**Repository:** `ChakrView`  
**Location:** `experiments/batch821_860/reports/REPRODUCIBILITY_REVIEW.md`

---

### 1. Cryptographic Lineage & Checkpoints

- **Canonical Checkpoint Step 10,000:**
  - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)
- **Batch 821–860 Pointer Candidates:**
  - Seed 42: `experiments/batch821_860/checkpoints/candidate_step300_seed42_pointer_planner.pt` (SHA-256: `95594aa712a2ca07b8a53c1f016334aa77ff0a2ba1a0c8b3e85292881bb595db`)
  - Seed 101: `experiments/batch821_860/checkpoints/candidate_step300_seed101_pointer_planner.pt` (SHA-256: `8c201639d6ec9908cf0d1645e3d74c0e64c398322ea3fe91a5dbfe0c325f54aa`)
  - Fixed-Gate Ablation: `experiments/batch821_860/checkpoints/candidate_step300_seed42_fixedgate_pointer_planner.pt` (SHA-256: `a93557457788b901a1cf4c4cb831fdbcfcf2ce611a134812a32cbe005b4104c8`)
- **Frozen Test Benchmark ($N=70$):**
  - File: `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
  - SHA-256: `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a` (Verified Untouched)

---

### 2. Exact Reproduction Commands

1. **Verify Canonical Hash:**
   ```powershell
   python -c "import hashlib; h = hashlib.sha256(open('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', 'rb').read()).hexdigest(); assert h == '96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee'; print('Canonical verified.')"
   ```

2. **Execute New Unit Tests:**
   ```powershell
   d:\Project\ChakrView\.venv_gpu\Scripts\python.exe -m pytest tests/test_batch821_860_pointer_generator.py tests/test_batch821_860_decoder_equivalence.py
   ```

3. **Train Pointer Candidates:**
   ```powershell
   $env:PYTHONPATH="."; d:\Project\ChakrView\.venv_gpu\Scripts\python.exe experiments/batch821_860/train_pointer_planner.py
   ```

4. **Execute Combined Generalization & Causal Grounding Evaluations:**
   ```powershell
   $env:PYTHONPATH="."; d:\Project\ChakrView\.venv_gpu\Scripts\python.exe experiments/batch821_860/evaluate_combined_pointer_generalization.py
   ```

5. **Run Full Regression Suite across Batches 541–860:**
   ```powershell
   d:\Project\ChakrView\.venv_gpu\Scripts\python.exe -m pytest tests/test_batch821_860_pointer_generator.py tests/test_batch821_860_decoder_equivalence.py tests/test_batch781_820_grammar_decoder.py tests/test_batch741_780_plan_dsl.py tests/test_batch701_740_neural_decomposition.py tests/test_batch661_700_dialogue_planning.py tests/test_batch621_660_multiturn_context.py tests/test_batch581_620_span_pointer.py tests/test_batch541_580_neural_intent.py
   ```
