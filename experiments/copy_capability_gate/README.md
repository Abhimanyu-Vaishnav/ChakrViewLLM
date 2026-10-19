# Copy Capability Gate (Inference-Only)

## Overview
This directory contains the experimental artifacts, execution runner, and diagnostic evaluation measuring whether dialog-style prompt formatting improves context-grounded copying in ChakrView independently of JSON syntax and constrained decoding.

## Safety & Non-Negotiable Boundaries
- **Zero Training**: No backward passes, optimizer steps, fine-tuning, or parameter updates.
- **Runtime Preservation**: `chakrview/runtime/session.py` is strictly preserved with zero modifications.
- **Canonical Immutability**: Step-10,000 checkpoint and tokenizer artifacts are cryptographically verified bit-identical.
- **Test Set Protection**: Held-out test split was never accessed.
- **Isolated Artifacts**: All runner scripts, data results, and documentation are strictly confined to `experiments/copy_capability_gate/`.

## Key Deliverables
- `COPY_CAPABILITY_GATE_REPORT.md`: Comprehensive analytical report detailing experimental setup, prompt framing comparisons, per-example metrics, root-cause findings, and final verdict.
- `COPY_CAPABILITY_RESULTS.json`: Complete empirical dataset recording exact prompt strings, pre-generation boundary token IDs, generated texts, expected ground truth, and per-example scores across all 20 validation passages and 3 prompt formats.
- `run_copy_capability_gate.py`: Fully reproducible inference-only evaluation harness.
- `test_copy_capability_gate.py`: Test suite validating cryptographic hashes, data integrity, passage ground-truth containment, and output structure.

## Reproduction Commands

To re-run the inference evaluation across all 20 validation passages:
```powershell
.venv\Scripts\python experiments/copy_capability_gate/run_copy_capability_gate.py
```

To run the test suite:
```powershell
python -m unittest tests/test_copy_capability_gate.py
```
