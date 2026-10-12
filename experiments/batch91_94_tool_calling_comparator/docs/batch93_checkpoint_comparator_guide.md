# Batch 93: Local Workbench Checkpoint Output Comparator Guide

## 1. Feature Overview
The Local Workbench Checkpoint Output Comparator allows researchers and ML engineers to compare outputs side-by-side from two checkpoints on a fixed set of evaluation prompts.

### Core Capabilities
1. **Checkpoint Identity & Provenance**: Computes SHA-256 digests, parameter counts (deduplicating tied embeddings), step counts, and condition metadata.
2. **Side-by-Side Evaluation**: Runs greedy deterministic decoding with identical tokenizers and decoding limits.
3. **Difference Highlighting**: Calculates unified diffs between checkpoint outputs and tags cases as `IDENTICAL` or `DIVERGENT`.
4. **Deterministic Oracle Integration**: Displays mathematical oracle reference answers and expected structured tool calls alongside model outputs.
5. **Persistence & Reopening**: Saves timestamped JSON comparison reports to `experiments/batch91_94_tool_calling_comparator/workbench/reports/` and allows instant reloading without rerunning inference.
6. **Robust Error Handling**: Missing or corrupted checkpoints fail gracefully without crashing the UI or CLI.

## 2. CLI Usage Examples

### Inspect a Checkpoint
```bash
python experiments/batch91_94_tool_calling_comparator/workbench/app.py inspect \
    --ckpt experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt
```

### Run Side-by-Side Checkpoint Comparison
```bash
python experiments/batch91_94_tool_calling_comparator/workbench/app.py compare \
    --ckpt-a experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt \
    --ckpt-b experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt \
    --max-tokens 40
```

### List Saved Comparison Reports
```bash
python experiments/batch91_94_tool_calling_comparator/workbench/app.py list-reports
```

### Launch Interactive Web Server
```bash
python experiments/batch91_94_tool_calling_comparator/workbench/app.py serve --port 8501
```
Open `http://localhost:8501` in any browser to use the graphical comparison interface.
