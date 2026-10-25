# ChakrView v0.8.0 Reproducibility & Evaluation Guide

This guide provides exact commands and deterministic configurations to reproduce all canonical evaluations and benchmark figures reported for ChakrView v0.8.0.

---

## 1. Environment & Determinism Flags

ChakrView evaluation is 100% CPU-deterministic under greedy decoding ($T=0.0$).

To configure strict reproducible execution in Python:
```python
import os
import torch
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)
np.random.seed(42)
```

---

## 2. Checkpoint & Tokenizer Pre-Flight Verification

Execute the following standalone command to verify bit-exact SHA-256 hashes:

```powershell
python -c "import hashlib, sys; \
files = [ \
  ('experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt', '96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee'), \
  ('data/experiments/vocab_4096/vocab.json', 'db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a'), \
  ('data/experiments/vocab_4096/merges.json', 'ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7'), \
  ('chakrview/brain/model.py', '1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361'), \
  ('chakrview/runtime/session.py', '9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67') \
]; \
all_pass = all(hashlib.sha256(open(p, 'rb').read()).hexdigest() == exp for p, exp in files); \
print('ALL CANONICAL HASHES VERIFIED: PASS' if all_pass else 'HASH MISMATCH DETECTED: FAIL'); \
sys.exit(0 if all_pass else 1)"
```

---

## 3. Sandboxed Coding Benchmark Execution

To reproduce the 6-category coding capability evaluation reported for the Step 10,000 base core:

```powershell
python -m evaluation.coding_suite.run_benchmark --output-dir evaluation/coding_suite/results
```

Expected baseline output for the 3.44M base pretraining core:
- **Total tasks**: 6
- **Test pass rate**: 1 / 6 (16.7%)
- **Syntax validity**: 3 / 6 (50.0%)
- **CPU throughput**: > 100 tokens/sec
- **Security / sandbox violations**: 0

---

## 4. Smoke Suite Reproduction

To run the automated regression smoke tests:

```powershell
pytest tests/test_v060_release_smoke.py -v
```
All 7 tests will pass in ~2 seconds.
