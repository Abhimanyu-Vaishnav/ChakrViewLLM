# ChakrView v0.8.0 Installation & Fresh-User Setup Guide

This guide describes how to set up ChakrView v0.8.0 from scratch on a clean system using pure CPU execution.

---

## 1. System Requirements

- **OS**: Windows 10/11 (PowerShell), Linux (Ubuntu 20.04+), or macOS
- **Python**: Python 3.10 – 3.14 (64-bit)
- **RAM**: Minimum 4 GB RAM (8 GB recommended)
- **Disk Space**: 500 MB free space (including checkpoint and virtual environment)
- **Hardware Acceleration**: None required. 100% CPU execution.

---

## 2. Step-by-Step Setup Instructions

### Step 1: Clone Repository
```powershell
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM
```

### Step 2: Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
Always quote package specifications to prevent shell redirection or wildcard expansion:
```powershell
python -m pip install --upgrade pip
python -m pip install "torch>=2.0.0" "numpy>=1.24.0" "pytest>=7.0.0" "cryptography>=40.0.0" "psutil>=5.9.0"
```

---

## 3. Checkpoint Download & Verification

The canonical checkpoint `training_checkpoint_step_10000.pt` (42,212,787 bytes) is distributed as a GitHub Release Asset attached to release `v0.8.0`.

### PowerShell Download & Pre-Flight Checksum:
```powershell
$targetDir = "experiments/batch16_21_combined/main_training_10000"
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
$targetFile = "$targetDir/training_checkpoint_step_10000.pt"

# Download the canonical release asset
$url = "https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/releases/download/v0.8.0/training_checkpoint_step_10000.pt"
Invoke-WebRequest -Uri $url -OutFile $targetFile

# Verify exact SHA-256
$expectedHash = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
$actualHash = (Get-FileHash -Algorithm SHA256 -Path $targetFile).Hash.ToLower()

if ($actualHash -eq $expectedHash) {
    Write-Host "[OK] Canonical checkpoint SHA-256 verified." -ForegroundColor Green
} else {
    Write-Error "[FAIL] Checksum mismatch! Expected $expectedHash, got $actualHash"
}
```

---

## 4. Run Verification Smoke Test

Run the automated release smoke test to confirm model architecture, tokenizer integrity, and deterministic CPU generation:

```powershell
pytest tests/test_v060_release_smoke.py -v
```

Expected output:
```text
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_01_canonical_checkpoint_hash_integrity PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_02_canonical_tokenizer_hashes_integrity PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_03_protected_session_py_unmodified PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_04_model_loading_and_parameter_shapes PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_05_deterministic_smoke_generation PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_06_unauthorized_checkpoint_fails_safely PASSED
tests/test_v060_release_smoke.py::TestV060ReleaseSmoke::test_07_invalid_sampling_parameters_fail_safely PASSED
============================== 7 passed in 2.22s ==============================
```

---

## 5. Minimal CPU Inference Example

```python
import torch
from pathlib import Path
from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter

# Initialize adapter (verifies canonical hashes automatically)
adapter = NeuralRuntimeAdapter(
    checkpoint_path=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
    tokenizer_dir=Path("data/experiments/vocab_4096"),
    device="cpu"
)

# Deterministic greedy generation
prompt = "The fundamental law of"
response = adapter.generate(prompt, max_new_tokens=16, temperature=0.0)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
```
