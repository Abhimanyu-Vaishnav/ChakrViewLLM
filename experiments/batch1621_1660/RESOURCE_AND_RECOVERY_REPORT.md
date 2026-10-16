# RESOURCE & RECOVERY REPORT: HARDWARE DISCIPLINE
## Batches 1621–1660 Hardware Execution Audit

### 1. Hardware Environment Specification

All benchmarks and training experiments for Batches 1621–1660 were designed and executed strictly within the local host hardware constraints:

- **Host OS:** Windows 11 (64-bit)
- **CPU:** Intel(R) Core(TM) i5-12450HX (12 logical processors, ~2.40 GHz base)
- **Physical RAM:** 15.69 GB Available (Usage maintained below 1 GB throughout execution)
- **GPU Usage:** None required; 100% PyTorch CPU execution.
- **Python Runtime:** Python 3.11.9 Virtual Environment (`.venv`)
- **PyTorch Version:** 2.5.1+cpu

---

### 2. Execution Resource Profiling

| Experiment Phase | Wallclock Time | Peak RSS Memory | Device | Process Count | Exit Code |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset Generation & Hashing** | 0.12 s | ~45 MB | CPU | 1 | 0 |
| **Canonical Checkpoint Evaluation (150 tasks)** | 18.47 s | ~120 MB | CPU | 1 | 0 |
| **Adapted Checkpoint Evaluation (150 tasks)** | 12.65 s | ~120 MB | CPU | 1 | 0 |
| **Candidate 100-Step Training Run** | 3.94 s | ~165 MB | CPU | 1 | 0 |
| **Candidate Checkpoint Evaluation (150 tasks)** | 12.41 s | ~120 MB | CPU | 1 | 0 |
| **Total Experiment Pipeline** | **47.59 s** | **< 200 MB** | **CPU** | **1** | **0** |

---

### 3. Stability, Resumability, and Recovery Discipline

1. **Deterministic Reproducibility:** All pseudorandom generators (PyTorch, Python `random`) were locked with seed `42`. Re-running the pipeline produces bit-exact weights, loss numbers, and generated strings.
2. **Crash & Leak Prevention:**
   - Evaluator runs in `torch.no_grad()` contexts to prevent memory accumulation.
   - Per-task generation limits token expansion to 16 tokens with strict EOS detection (`eos_token_id=3`).
   - Checkpoint integrity is validated pre- and post-flight against immutable SHA-256 digests.
3. **Interrupt Recovery:** The training script produces standalone, state-complete `.pt` checkpoints containing `step`, `model_state_dict`, `optimizer_state_dict`, and `val_loss`. Any interrupted run can resume directly from the saved step state.
4. **Machine Preservation:** No system-level environment variables were altered, no machine-level software was installed, and no unapproved external dependencies were added.
