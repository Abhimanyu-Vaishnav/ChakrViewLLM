# Reproducibility Review: Batches 1021–1060
## Local Knowledge Assistant MVP

### 1. Invariant Integrity Verification
- **Branch:** `dev/v0.6-v0.7-validation`
- **Canonical Checkpoint Path:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Observed SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Integrity Status:** Strictly Verified and Unmodified.

### 2. Environment Details
- **Operating System:** Windows 11 Enterprise (amd64)
- **Python Version:** Python 3.14.7 (Virtualenv: `.\.venv\Scripts\python.exe`)
- **Key Dependencies:** PyTorch LTS, NumPy, PyTest
- **Hardware Profile:** Multi-core CPU with AVX2 instruction set

### 3. Execution Commands to Reproduce
1. **Unit & Integration Tests:**
   ```powershell
   $env:PYTHONPATH="."
   .\.venv\Scripts\python.exe -m pytest tests/test_batch1021_1060_local_assistant.py -v
   ```
   *Expected Outcome:* 4 passed in ~1.9s.

2. **Frozen 50-Item Benchmark Evaluation:**
   ```powershell
   $env:PYTHONPATH="."
   .\.venv\Scripts\python.exe experiments/batch1021_1060/run_evaluation.py
   ```
   *Expected Outcome:* 50/50 passed (100.0%), Median latency 0.46 ms, p95 latency 0.87 ms.

3. **CLI Interface Validation:**
   ```powershell
   $env:PYTHONPATH="."
   .\.venv\Scripts\python.exe -m chakrview.runtime.assistant_cli list-docs
   .\.venv\Scripts\python.exe -m chakrview.runtime.assistant_cli ask "What is the context window ceiling of ChakrMicro?"
   .\.venv\Scripts\python.exe -m chakrview.runtime.assistant_cli ask "What was the GDP of France in 2023?"
   ```

4. **Historical Regression Suites:**
   ```powershell
   $env:PYTHONPATH="."
   .\.venv\Scripts\python.exe -m pytest tests/test_batch143_146_instruction_regression.py tests/test_batch861_900_span_grounding.py -v
   ```
   *Expected Outcome:* 8 passed in ~2.0s.
