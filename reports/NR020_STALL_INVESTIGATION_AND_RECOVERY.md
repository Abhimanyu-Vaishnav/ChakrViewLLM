# NR-020 Evaluation Stall Investigation & Recovery Report

## 1. Executive Summary & Diagnostic Verdict

During the execution of CV-GENESIS-NR-020 Phase 6 (Evaluation of the Fresh 1,000-Task Strictly Matched Benchmark), background task `task-3718` (PID 1564) ran for approximately 66 minutes (~4,082s CPU time / 10,130s total thread processor time) without writing output files or emitting terminal progress.

A comprehensive forensic inspection was performed pursuant to the user directive.

### Diagnostic Verdict:
1. **Root Cause:** A deterministic loop stall in the benchmark generation function `build_1000_matched_benchmark()`. Specifically, the generator enforced strict global uniqueness across all 1,000 prompts while drawing from limited word and variable pools (9 unique words and 5 variable names) without parameter salting for operations `string_upper`, `string_lower`, `string_reverse`, `string_length`, and `var_set`. For example, `string_upper` across 3 languages could only ever generate at most 9 distinct prompts; when attempting to draw 25 unique samples in a `while len < 25:` loop, the process entered an infinite retry loop with 100% CPU thread burn.
2. **Process Disposition:** PID 1564 was safely terminated. No unrelated Python processes, canonical checkpoints, or historical artifacts were affected.
3. **Partial Results Audit:** Because the stall occurred in the generation phase before the evaluation loop initialized, 0 benchmark tasks had been evaluated, and `data/stage_j_020/matched_benchmark_1000_results.json` had not yet been created. No historical data was lost or overwritten.
4. **Architectural Fix:** Expanded vocabulary and variable pools with index salting ensuring mathematical guarantee of 100% prompt uniqueness (1,000 unique prompts generated in 0.12 seconds). Implemented per-batch incremental disk serialization (`batch_size=10`) with full task-ID resumability and live ETA / throughput logging.
5. **Recovery Verification:** A 20-task diagnostic batch was executed on CUDA, confirming flawless execution at **2.31 tasks/second** (8.7 seconds total). The full 1,000-task recovery run was launched with an expected completion time of ~7.2 minutes.

---

## 2. Process & Hardware Inspection (PID 1564)

Prior to process termination, full process telemetry was gathered:

- **Process ID (PID):** `1564`
- **Parent Process ID:** `25544` (PowerShell agent host)
- **Command Line:** `"D:\Project\ChakrView\.venv_gpu\Scripts\python.exe" experiments/genesis_stage_j_020/generate_and_eval_1000_benchmark.py`
- **Working Set Memory:** 205.4 MB
- **User/Kernel CPU Time:** 10,130.8 seconds (2 hours 48 minutes total processor time across threads)
- **Active Threads:** 28 threads (1 actively spinning in user space, 27 waiting)
- **GPU Utilization:** 0%
- **GPU VRAM Allocation:** 0 MiB / 6,144 MiB (CUDA was never reached or initialized)
- **Execution Location:** Caught inside `build_1000_matched_benchmark()` while attempting `seen_prompts.add(s["prompt"])`.

---

## 3. Bottleneck Analysis & Evaluation Optimization

### Identified Flaws in Original Implementation:
1. **Unbounded Generation Loop:** `while op_count < 25:` looped infinitely because the generator's underlying dictionary could not produce 25 unique prompts for unary string operations without parameter permutation.
2. **Delayed Disk Serialization:** The original script held all results in memory and only attempted to write `matched_benchmark_1000_results.json` at the very end of task 1,000.
3. **Silent Execution:** Absence of intermediate stdout flushing or per-batch checkpoints.

### Applied Optimizations:
1. **Seeded Robust Generator:** Augmented `generate_robust_sample()` with 52 diverse lexical terms and 25 variable names plus index salts, enabling generation of 1,000 distinct prompts with 0 collisions in under 0.2s.
2. **Manifest Decoupling:** The benchmark manifest `data/stage_j_020/fresh_matched_benchmark_1000.json` is generated once and sealed to disk. Subsequent runs simply load the sealed JSON manifest.
3. **Resumable Batch Architecture:** Every 10 evaluated tasks, predictions and running accuracy metrics are flushed to disk. If interrupted, the script resumes exactly from the last completed `task_id`.
4. **Throughput Logging:** Displays elapsed time, processing rate (tasks/sec), and real-time ETA.

---

## 4. Diagnostic Batch Results (20 Tasks)

To verify the pipeline before the full run, a 20-task diagnostic batch was evaluated on CUDA:
- **Elapsed Time:** 8.7 seconds
- **Throughput:** 2.31 tasks/second
- **VRAM Utilization:** Steady at ~2.1 GB
- **System Accuracies on Diagnostic Split (20 ID tasks):**
  - **System A (Pure Neural):** 0.0% (0/20)
  - **System B (Deterministic Parser):** 25.0% (5/20)
  - **System C (Neural Router Direct):** 0.0% (0/20)
  - **System D (Neural Source-Grounded):** 25.0% (5/20)
  - **System E (Cascaded Hybrid):** 30.0% (6/20)

---

## 5. Artifact & Integrity Ledger

| Artifact Path | Description | Status |
| :--- | :--- | :--- |
| `data/stage_j_020/fresh_matched_benchmark_1000.json` | 1,000 strictly matched ID/OOD tasks (100% unique prompts) | Sealed |
| `data/stage_j_020/diagnostic_20_results.json` | 20-task diagnostic verification run results | Verified |
| `experiments/genesis_stage_j_020/generate_and_eval_1000_benchmark.py` | Optimized resumable evaluation runner | Active |
| `reports/NR020_STALL_INVESTIGATION_AND_RECOVERY.md` | Forensic audit and recovery report | Authored |

Canonical checkpoints (`genesis_17m_staged_best.pt`, SHA-256 `2c21632e...`) remain untouched and sealed.
