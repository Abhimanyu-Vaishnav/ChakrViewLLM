# ChakrView v0.7.0 — Reproducibility & Verification Guide

This guide details the exact steps to reproduce test suites, benchmarks, and CLI commands in ChakrView v0.7.0.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.14.7)
- PyTorch 2.0+ (Tested on PyTorch 2.14.0+cpu)
- Operating System: Windows 11, Linux, or macOS

### Installation
```bash
# Clone the repository
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Running Regression Tests

Execute all 38 automated regression tests across runtime, security, benchmarks, and data pipelines:

```bash
# Set PYTHONPATH to project root
export PYTHONPATH="."   # On Linux/macOS
$env:PYTHONPATH="."     # On Windows PowerShell

pytest tests/test_coding_data_pipeline.py \
       tests/test_security_hardened_executor.py \
       tests/test_coding_suite.py \
       tests/test_coding_assistant.py \
       tests/test_inference_session.py \
       tests/test_v060_release_smoke.py \
       tests/test_intelligence_suite.py \
       tests/test_sampling_ngram_blocking.py \
       tests/test_intelligence_harness_audit.py -v
```

Expected result: **38 passed, 3 skipped, 0 failed**.

---

## 3. Running Real Native Model Benchmarks

### 10-Category 32-Task Intelligence Benchmark
Point to the canonical Step 10,000 checkpoint:

```bash
# Run baseline greedy decoding
python evaluation/intelligence_suite/run_intelligence_benchmark.py \
  --repetition-penalty 1.0 \
  --no-repeat-ngram-size 0 \
  --output evaluation/intelligence_suite/run_greedy_baseline.json

# Run with sliding-window n-gram repetition blocking (n=3)
python evaluation/intelligence_suite/run_intelligence_benchmark.py \
  --repetition-penalty 1.0 \
  --no-repeat-ngram-size 3 \
  --output evaluation/intelligence_suite/run_ngram3.json
```

### 12-Task Coding Suite Evaluation
```bash
python evaluation/coding_suite/run_coding_evaluation.py
```
Expected result: **Success: 2/12 (16.7%) | Syntax: 5/12 (41.7%)**.

---

## 4. CLI Usage Examples

```bash
# Check hardware topology and runtime status
python -m chakrview.runtime.assistant_cli status

# Query the local knowledge assistant
python -m chakrview.runtime.assistant_cli ask "What is ChakrView?"

# Draft code using the coding assistant
python -m chakrview.runtime.assistant_cli code draft "def multiply(a, b):"

# Execute sandboxed test assertion fixture
python -m chakrview.runtime.assistant_cli code sandbox "def add(a, b): return a + b" --test-assertions "assert add(2, 3) == 5"
```