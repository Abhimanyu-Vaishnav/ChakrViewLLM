# SFT V3 Failure Analysis & Structured Extraction Recovery

## Overview
This directory contains the root-cause analysis, metric reconciliation, inference-only diagnostic suite, and progressive extraction evaluation investigating why the 100-step SFT V3 pilot reduced validation loss from 5.3873 to 3.6204 while achieving 0/30 exact task matches and developing a strong Python doctest attractor (`{"__main__":\n import doctest`).

## Absolute Safety & Reproducibility Constraints
- **Zero Training**: No optimizer steps, backward passes, or parameter modifications were executed.
- **Untouched Session**: `chakrview/runtime/session.py` was strictly preserved with 0 diffs.
- **Canonical Checkpoint Intact**: `training_checkpoint_step_10000.pt` and tokenizer vocab/merges remain cryptographically identical.
- **Test Set Protection**: `test.jsonl` was never accessed or inspected.

## Key Files
- `SFT_V3_ROOT_CAUSE_REPORT.md`: Comprehensive executive report synthesizing all findings and root causes.
- `PILOT_METRIC_RECONCILIATION.md`: Mathematical reconciliation of candidate validation loss (3.6204 full vs 3.3215 5-batch) and evaluation scoring details.
- `PROMPT_ABLATION_PROTOCOL.md`: Protocol detailing the prompt delimiter and framing variations tested.
- `ROOT_CAUSE_RANKING.json`: Structured ranking of the 8 candidate hypotheses by evidence strength and confidence.
- `EXTRACTION_DIAGNOSTICS.json`: Raw empirical outputs from the inference diagnostic runs (Prompt ablations, Tasks A–E, and Trie reachability).
- `run_inference_diagnostics.py`: Reproducible inference-only runner executing prompt ablations and progressive extraction tasks on held-out validation records.
- `test_sft_v3_failure_analysis.py`: Unit and regression test suite verifying metric reconciliation logic, prompt template constructors, and trie reachability.

## Reproducing Diagnostics
To execute the inference-only diagnostic suite:
```powershell
python experiments/sft_v3_failure_analysis/run_inference_diagnostics.py
```

To execute the test suite:
```powershell
python -m unittest tests/test_sft_v3_failure_analysis.py
```
