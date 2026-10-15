# Reproducibility Review: Batches 781–820
## Exact Commands, Environment Lineage, Checkpoint Hashes & Verification

**Date:** 2026-10-15  
**Milestone:** Batches 781–820  
**Repository:** `ChakrView`  
**Location:** `experiments/batch781_820/reports/REPRODUCIBILITY_REVIEW.md`

---

### 1. Environment & Hardware Specifications

- **Operating System:** Windows 11
- **Python Executable:** `D:\Project\ChakrView\.venv_gpu\Scripts\python.exe`
- **PyTorch Version:** 2.14.1+cu126 (CUDA 12.6)
- **GPU Device:** NVIDIA RTX A1000 6GB Laptop GPU
- **Tokenizer:** BPE Tokenizer (vocab size 4,096) at `release/chakrview-v0.5.0/data/experiments/vocab_4096`

---

### 2. Checkpoint & Dataset Cryptographic Hashes

- **Canonical Checkpoint Step 10,000:**
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)
- **Batch 781–820 Candidate (Seed 42):**
  - Path: `experiments/batch781_820/checkpoints/candidate_step300_seed42_grammar_planner.pt`
  - SHA-256: `e3f2fbb2d2e0aa82992daeb48e821953d69d9404cfd13f0d95e0ddd9478ed4e8`
- **Batch 781–820 Candidate (Seed 101):**
  - Path: `experiments/batch781_820/checkpoints/candidate_step300_seed101_grammar_planner.pt`
  - SHA-256: `74cdb1403673c9680e1ae5c261ed218726c1675fa3ffb495e47498e159408837`
- **Frozen Test Dataset (N=70):**
  - Path: `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
  - SHA-256: `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a`

---

### 3. Exact Commands to Reproduce Milestone Results

1. **Verify Canonical Checkpoint Integrity:**
   ```powershell
   python -c "import hashlib; h = hashlib.sha256(open('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', 'rb').read()).hexdigest(); assert h == '96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee'; print('Canonical checkpoint verified intact.')"
   ```

2. **Execute Unit Tests for Grammar Decoder:**
   ```powershell
   d:\Project\ChakrView\.venv_gpu\Scripts\python.exe -m pytest tests/test_batch781_820_grammar_decoder.py
   ```

3. **Generate Dataset and Verify Zero Prompt Leakage:**
   ```powershell
   $env:PYTHONPATH="."; d:\Project\ChakrView\.venv_gpu\Scripts\python.exe experiments/batch781_820/build_expanded_plan_dsl_dataset.py
   ```

4. **Run Generalization & Safety Evaluations:**
   ```powershell
   $env:PYTHONPATH="."; d:\Project\ChakrView\.venv_gpu\Scripts\python.exe experiments/batch781_820/evaluate_generalization_and_safety.py
   ```

5. **Run Argument Fidelity Benchmark:**
   ```powershell
   $env:PYTHONPATH="."; d:\Project\ChakrView\.venv_gpu\Scripts\python.exe experiments/batch781_820/evaluate_argument_fidelity.py
   ```

6. **Run Full 40-Item Regression Suite across Batches 541–820:**
   ```powershell
   d:\Project\ChakrView\.venv_gpu\Scripts\python.exe -m pytest tests/test_batch781_820_grammar_decoder.py tests/test_batch741_780_plan_dsl.py tests/test_batch701_740_neural_decomposition.py tests/test_batch661_700_dialogue_planning.py tests/test_batch621_660_multiturn_context.py tests/test_batch581_620_span_pointer.py tests/test_batch541_580_neural_intent.py
   ```
