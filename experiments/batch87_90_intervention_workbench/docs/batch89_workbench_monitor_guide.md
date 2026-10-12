# Batch 89 — Enhanced Local Workbench & Training Monitor Guide

## 1. Product Enhancements in v0.2

The **ChakrView Local Workbench** was upgraded to v0.2 with real-time asynchronous training monitoring and loss plotting capabilities:

Location: [`experiments/batch87_90_intervention_workbench/workbench/app.py`](file:///d:/Project/ChakrView/experiments/batch87_90_intervention_workbench/workbench/app.py)

---

## 2. New Features

1. **Live Training Job Monitor**:
   - Launches asynchronous training jobs in background threads without blocking or freezing the web server.
   - Real-time step progress bar and status indicator (`RUNNING`, `COMPLETED`, `FAILED`).
2. **HTML5 Canvas Real-Time Loss Plotter**:
   - Renders interactive cross-entropy loss descent curves in real time directly on the dashboard without external JavaScript libraries.
3. **Automatic Report Persistence**:
   - Training runs automatically persist structured JSON reports with full step-by-step loss history into `workbench/reports/`.
4. **Historical Report Reopening**:
   - Inspect and review previous sprint summaries and reports without needing to retrain.
5. **Deterministic Oracle Distinction**:
   - Retains clear visual tagging: `DETERMINISTIC_ORACLE` vs `CHAKRMICRO_LOCAL_MODEL` to prevent conflating software logic with neural capability.

---

## 3. How to Launch and Use the Enhanced Workbench

### Launch Web Server:
```powershell
.\.venv\Scripts\python.exe experiments/batch87_90_intervention_workbench/workbench/app.py serve --port 8501
```
Open browser at `http://127.0.0.1:8501`.

### Use CLI Subcommands:
```powershell
# Inspect Checkpoint
.\.venv\Scripts\python.exe experiments/batch87_90_intervention_workbench/workbench/app.py inspect --path experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt

# Run Deterministic Oracle
.\.venv\Scripts\python.exe experiments/batch87_90_intervention_workbench/workbench/app.py oracle --a 3 --b 4 --mode scratchpad

# Run Local Model Inference
.\.venv\Scripts\python.exe experiments/batch87_90_intervention_workbench/workbench/app.py infer --path experiments/batch87_90_intervention_workbench/experiments/condition_b_incremental_step_100.pt --prompt "<|im_start|>user`nAdd: 3 + 4 = ?<|im_end|>`n<|im_start|>assistant`n"
```
