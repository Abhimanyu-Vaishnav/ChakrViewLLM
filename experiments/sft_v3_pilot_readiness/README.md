# ChakrView SFT V3 Training Compatibility, Dry-Run & Pilot Readiness

This directory contains the dry-run compatibility harness, pipeline trace reports, replay data audits, and candidate pilot protocols for SFT V3.

## Directory Structure & Deliverables

- `README.md`: Overview, reproduction commands, safety boundaries, and manifest summary.
- `SFT_V3_PIPELINE_COMPATIBILITY_REPORT.md`: Step-by-step trace from JSONL storage to tensor collation and model forward loss.
- `REPLAY_DATA_AUDIT.md`: Analysis of replay provenance, document isolation, causal token shift, and mixture policy trade-offs.
- `SFT_V3_PILOT_PROTOCOL.md`: Exact candidate configuration, evaluation plan across 3 capability layers, and explicit success/failure criteria.
- `DRY_RUN_RESULTS.json`: Machine-readable results from the forward-only compatibility execution.
- `PILOT_READINESS_MANIFEST.json`: Cryptographic hashes, configuration parameters, and safety gate confirmations.
- `run_sft_v3_dry_run.py`: Safe, isolated forward-only verification script.

## Safe Reproduction Commands

To independently reproduce the compatibility checks and regression test suites:

```powershell
# 1. Run the safe, forward-only dry-run harness (computes forward loss without backward/optimizer)
.\.venv\Scripts\python.exe experiments/sft_v3_pilot_readiness/run_sft_v3_dry_run.py

# 2. Run the deterministic unit test suite (11/11 tests passing)
.\.venv\Scripts\python.exe -m pytest tests/test_sft_v3_dataset_builder.py tests/test_schema_decoder_hardening.py
```

## Hard Safety Confirmation
- **ZERO** model training, fine-tuning, optimizer steps, or backward passes were executed.
- `chakrview/runtime/session.py` has **ZERO** modifications.
- Canonical checkpoints and tokenizers remain 100% bit-identical.
- Held-out test split (`test.jsonl`) was **NEVER** accessed or inspected.
- Checkpoints write to isolated experiment directories only upon separate, explicit authorization.
