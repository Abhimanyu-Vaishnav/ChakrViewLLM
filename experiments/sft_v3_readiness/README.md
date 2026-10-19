# ChakrView SFT V3 Readiness & Engineering Milestone

This directory contains the audit findings, diagnostic tools, curriculum specifications, and evaluation protocols prepared during the SFT V3 Readiness milestone.

## Directory Structure & Key Deliverables

- `SFT_V3_READINESS_REPORT.md`: Comprehensive executive report detailing verdict, root causes, comparative benchmarks, and safety verifications.
- `SFT_V3_CURRICULUM_SPEC.md`: Technical specification for the SFT V3 training curriculum, negative-example policy, and data quality requirements.
- `SFT_V3_EVALUATION_PROTOCOL.md`: Leakage-safe evaluation framework separating Native Generation (Layer A), Constrained Decoding (Layer B), and Semantic Task Performance (Layer C).
- `AUDIT_RESULTS.json`: Machine-readable results of the pipeline audit, token alignment checks, and 3-way comparative benchmarks.
- `BASELINE_INVENTORY.md`: Architectural baseline, checkpoint hashes, and pre-existing environment inventory.
- `DRAFT_SFT_V3_EXAMPLES.jsonl`: 10 grounded, synthetic curriculum examples demonstrating the SFT V3 specification.
- `audit_sft_pipeline.py`: Automated audit script analyzing split isolation, exact/fuzzy duplicate checks, and format distributions.
- `audit_token_alignment.py`: Diagnostic script verifying causal sequence alignment, loss-masking integrity, and truncation limits.
- `benchmark_comparative.py`: 3-way comparative evaluation harness measuring unconstrained vs constrained decoding on the validation split.
- `build_draft_curriculum.py`: Generator for draft curriculum examples.

## Verification & Safe Execution

To independently reproduce the safe diagnostic audits and test suites:

```powershell
# 1. Run the test suites (20 tests passed)
.\.venv\Scripts\python.exe -m pytest tests/test_json_constrained_decoder.py tests/test_schema_prefix_trie.py tests/test_fake_model_schema_decoder.py tests/test_schema_decoder_hardening.py

# 2. Re-run the data pipeline audit
.\.venv\Scripts\python.exe experiments/sft_v3_readiness/audit_sft_pipeline.py

# 3. Re-run token alignment and loss-masking verification
.\.venv\Scripts\python.exe experiments/sft_v3_readiness/audit_token_alignment.py

# 4. Re-run the 3-way comparative evaluation
.\.venv\Scripts\python.exe experiments/sft_v3_readiness/benchmark_comparative.py
```

## Hard Safety Confirmation
- **ZERO** model training, fine-tuning, backward passes, or optimizer steps were executed.
- `chakrview/runtime/session.py` has **ZERO** modifications.
- Canonical checkpoints and tokenizers remain 100% bit-identical.
- Held-out test split (`test.jsonl`) was **NEVER** accessed or inspected.
