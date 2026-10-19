# ChakrView v0.6.0 Environment Setup & Reproducibility Guide

This document provides step-by-step instructions to set up the local environment, verify artifact checksums, execute release smoke tests, and run inference benchmarks.

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **OS**: Windows 10/11 or modern Linux (Ubuntu 22.04+ recommended).
- **Python**: Python 3.10 – 3.14 (64-bit).
- **RAM**: Minimum 4 GB available system RAM.
- **Disk**: ~2 GB free disk space for repository, checkpoints, and dependencies.

### Environment Installation
From the repository root (`d:\Project\ChakrView`):
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install torch numpy pytest psutil cryptography
```

---

## 2. Cryptographic Checksum Verification

Verify the integrity of canonical model weights and tokenizer files:

```powershell
python -c "
import hashlib
files = {
    'Canonical Checkpoint': ('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', '96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee'),
    'Vocab': ('data/experiments/vocab_4096/vocab.json', 'db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a'),
    'Merges': ('data/experiments/vocab_4096/merges.json', 'ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7')
}
for name, (path, exp_hash) in files.items():
    with open(path, 'rb') as f:
        act_hash = hashlib.sha256(f.read()).hexdigest()
    assert act_hash == exp_hash, f'Mismatch in {name}'
    print(f'[PASS] {name} hash verified.')
"
```

---

## 3. Running Automated Smoke & Regression Tests

Execute the automated release test suite:
```powershell
.venv\Scripts\python -m unittest tests/test_v060_release_smoke.py
```

Expected output:
```text
.......
----------------------------------------------------------------------
Ran 7 tests in 0.54s

OK
```

Execute diagnostic regression tests:
```powershell
.venv\Scripts\python -m unittest tests/test_copy_capability_gate.py tests/test_sft_v3_failure_analysis.py
```

Expected output:
```text
.........
----------------------------------------------------------------------
Ran 9 tests in 0.01s

OK
```

---

## 4. Running the Quickstart Inference Demo

To run a minimal CPU generation pass with the canonical checkpoint:
```powershell
.venv\Scripts\python release/chakrview-v0.6.0/quickstart_inference.py
```

---

## 5. Running the Deterministic Benchmark

To reproduce latency, throughput, and memory measurements on your local machine:
```powershell
.venv\Scripts\python release/chakrview-v0.6.0/benchmark_v060.py
```

Results will be automatically updated in `release/chakrview-v0.6.0/BENCHMARK_RESULTS.json`.
