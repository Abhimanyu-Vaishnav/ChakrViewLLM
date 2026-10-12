# Batch 97: Workbench Token-Level Checkpoint Diagnostics Guide

## 1. Feature Architecture & Diagnostic Modalities
The enhanced Local Workbench includes token-level scoring and divergence tracing:
1. **Teacher-Forced Scoring**: Evaluates the log-probability assigned by each model to a fixed prompt prefix without generating new tokens.
2. **Free-Running Divergence Tracing**: Allows each model to generate independently from a prompt, records step-by-step token probabilities and top-K candidate tokens, and identifies the exact step where generation diverges.
3. **Perplexity Scoring**: Computes the sequence perplexity on each model's generated continuation.

## 2. CLI Usage Examples

### Inspect a Checkpoint
```bash
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py inspect \
    --ckpt experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt
```

### Run Token-Level Diagnostics
```bash
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py diagnose \
    --ckpt-a experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt \
    --ckpt-b experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt \
    --max-tokens 35
```

### List Persisted Reports
```bash
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py list-reports
```

### Launch Interactive Web Server
```bash
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py serve --port 8501
```
Open `http://localhost:8501` to view step-by-step token divergence cards, candidate probabilities, and comparative perplexity stats.
