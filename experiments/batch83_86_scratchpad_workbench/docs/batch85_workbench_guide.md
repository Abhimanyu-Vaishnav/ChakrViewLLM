# Batch 85 — ChakrView Local Workbench Product Specification and Guide

## 1. Product Overview

The **ChakrView Local Workbench** is a zero-dependency, local-first interactive development tool and CLI for inspecting ChakrMicro checkpoints, running local inferences, auditing model outputs against deterministic oracles, and reviewing historical sprint experiment records.

Location: [`experiments/batch83_86_scratchpad_workbench/workbench/app.py`](file:///d:/Project/ChakrView/experiments/batch83_86_scratchpad_workbench/workbench/app.py)

---

## 2. Key Product Capabilities

1. **Checkpoint Verification & Inspection**:
   - Computes SHA-256 digests, parameter counts, disk size, and metadata without modifying the checkpoint files.
2. **Local Model Inference Engine**:
   - Loads any ChakrMicro PyTorch checkpoint and runs CPU inference locally using the canonical BPE tokenizer.
3. **Deterministic Arithmetic Oracle Comparator**:
   - Provides ground-truth arithmetic evaluations (in direct and scratchpad formats) explicitly labeled as `DETERMINISTIC_ORACLE` so that software capabilities are never conflated with model intelligence.
4. **Historical Experiment & Report Viewer**:
   - Reopens saved experiment metrics, loss curves, and diagnostic tables without requiring retraining.
5. **Two Execution Modes**:
   - **CLI Mode**: Direct script execution for automated pipelines (`inspect`, `infer`, `oracle`, `serve`).
   - **Browser Web UI Mode**: Clean dark-mode dashboard running on `http://127.0.0.1:8501` built with pure standard-library HTML5/CSS/JavaScript (zero third-party dependencies required).

---

## 3. Launch and Usage Commands

### A. Launch Local Web Dashboard
```powershell
.\.venv\Scripts\python.exe experiments/batch83_86_scratchpad_workbench/workbench/app.py serve --port 8501
```
Open your browser at `http://127.0.0.1:8501`.

### B. Inspect a Checkpoint via CLI
```powershell
.\.venv\Scripts\python.exe experiments/batch83_86_scratchpad_workbench/workbench/app.py inspect --path experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt
```

### C. Run Deterministic Oracle via CLI
```powershell
.\.venv\Scripts\python.exe experiments/batch83_86_scratchpad_workbench/workbench/app.py oracle --a 4 --b 5 --mode scratchpad
```

### D. Run Model Inference via CLI
```powershell
.\.venv\Scripts\python.exe experiments/batch83_86_scratchpad_workbench/workbench/app.py infer --path experiments/batch83_86_scratchpad_workbench/experiments/condition_b_scratchpad_arithmetic_step_100.pt --prompt "<|im_start|>user`nAdd with scratchpad: 2 + 3 = ?<|im_end|>`n<|im_start|>assistant`n"
```
