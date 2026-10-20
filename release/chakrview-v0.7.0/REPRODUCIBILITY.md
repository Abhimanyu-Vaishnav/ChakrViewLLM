# ChakrView v0.7.0 Quickstart & Environment Reproducibility Guide

This guide describes how to set up the development environment, run the weight-independent test suite, execute the coding capability benchmark, and operate the coding assistant CLI.

---

## 1. Supported Platform & Python Environment

- **Operating System**: Windows 10/11 or modern Linux (Ubuntu 22.04+).
- **Python Version**: Python 3.10 – 3.14 (64-bit).
- **Hardware Architecture**: CPU-first, hardware-adaptive. No GPU requirement is imposed.

---

## 2. Virtual Environment Setup & Dependencies

From the repository root (D:\Project\ChakrView-v07-coding-foundation):

`powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
`

Tracked dependencies declared in equirements.txt:
- 
umpy==2.5.3: Multi-dimensional tensor mathematics
- pytest==9.1.1: Automated test execution
- cryptography>=42.0.0: Secure transport and key hashing
- 	orch>=2.0.0: Neural runtime model execution (CPU-compatible)
- psutil>=5.9.0: Process telemetry and resource monitoring

---

## 3. Running Weight-Independent Tests

All core coding foundation components (synthetic data pipeline, restricted test runner, coding assistant validation, and task suite schema) run completely independently of neural model weights:

`powershell
# Run the entire test suite
pytest tests/test_coding_data_pipeline.py 
       tests/test_security_hardened_executor.py 
       tests/test_coding_suite.py 
       tests/test_coding_assistant.py 
       tests/test_inference_session.py 
       tests/test_batch2221_2260_cli_and_integration.py 
       tests/test_v060_release_smoke.py -v
`

Expected Outcome: **33 passed, 3 skipped, 0 failed** in ~20s.

---

## 4. Checkpoint Availability & Smoke-Test Behavior

The canonical Step-10,000 checkpoint (	raining_checkpoint_step_10000.pt, SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee) is gitignored by project policy (*.pt).

- **When the checkpoint is absent**:
  - The 3 model-dependent tests in 	est_v060_release_smoke.py (checksum verification, model shape loading, and deterministic smoke text generation) skip gracefully with explicit diagnostic reasons.
  - They do **not** report as passed.
  - All other 4 tests in 	est_v060_release_smoke.py (tokenizer hashes, session file protection, unauthorized checkpoint rejection, sampling parameter validation) run and pass unconditionally.
- **Supplying an Authorized Checkpoint**:
  - If you have an authorized copy of the canonical checkpoint, place it at:
    experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt
  - All 7 tests in 	est_v060_release_smoke.py will automatically execute and verify SHA-256 integrity.
  - Do not commit or publish .pt files.

---

## 5. Running the Coding Capability Benchmark

The benchmark evaluates Python coding performance across 12 standardized tasks covering 6 core software development categories.

### Deterministic Smoke Mode (Weight-Independent)
Runs task loading, schema validation, and process isolation without requiring neural weights:
`powershell
python evaluation/coding_suite/run_coding_evaluation.py --smoke --output-path evaluation/coding_suite/smoke_results.json
`

### Full Neural Evaluation (Requires Checkpoint)
When the canonical checkpoint is present:
`powershell
python evaluation/coding_suite/run_coding_evaluation.py
`
Results are saved to evaluation/coding_suite/coding_baseline_results.json.

---

## 6. Distinguishing Model Intelligence vs Tool-Assisted Execution

ChakrView strictly demarcates neural generation from deterministic tooling:

| Command / Component | Execution Mode | Underlying Mechanism | Classification |
| :--- | :--- | :--- | :--- |
| ssistant_cli status | Tool | System query (psutil, platform) | Hardware Telemetry |
| ssistant_cli ask "What is 15 + 27?" | Tool Routing | Python AST safe calculator | AST Calculator (Deterministic) |
| ssistant_cli code sandbox "..." | Tool-Assisted | Child process via isolated_executor.py | Subprocess Test Runner |
| ssistant_cli code draft "..." | Native Neural | Autoregressive ChakrMicro core | Model Generation |

---

## 7. Security Architecture of Restricted Runner

evaluation/coding_suite/isolated_executor.py executes code in a dedicated child process with:
1. Strict static AST pre-auditing rejecting all imports, dynamic code execution (eval/exec), and introspection attributes (__subclasses__, __globals__).
2. Dual verification of both code payload and test assertions.
3. Hard subprocess wall-clock timeouts (default: 2.0s).
4. Sanitized environment variables and clean temporary directories.
5. JSON file payload serialization to prevent string-interpolation escaping.

**Security Boundary**: This is a restricted local test runner for functional testing. It is **not** an operating-system kernel container (e.g. gVisor, Docker jail). Untrusted public code must be run inside dedicated container/VM infrastructure.