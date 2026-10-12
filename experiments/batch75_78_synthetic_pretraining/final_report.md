# ChakrView Master Sprint Batches 75–78: Final Report

## Executive Summary

Sprint Batches 75–78 tested whether controlled synthetic mathematical pretraining improves the arithmetic reasoning and held-out generalization of **ChakrMicro v0.1** (3,443,136 parameters), or whether the model remains constrained to memorization and modal collapse.

### Headline Results:
1. **Pipeline & Invariant Audit**:
   - The canonical architecture (6 layers, $d_{model}=192$, $n_{heads}=6$, $hidden\_dim=512$, $vocab\_size=4096$, $max\_seq\_len=512$) and step-10,000 checkpoint SHA-256 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) were verified bit-for-bit.
   - The uncommitted user modification in `chakrview/runtime/session.py` remained preserved and untouched.
   - All 101 existing regression tests from Batches 33–74 passed completely.
2. **Controlled Experimental Design**:
   - Condition B (Synthetic Arithmetic Curriculum) generated 56,231 tokens from 8 diverse arithmetic templates drawing exclusively from the 30 allowed training operand pairs from Batch 72, with 0 split leakage into the 10 validation pairs or 10 held-out test pairs.
   - Condition A (Matched Language Control) was constructed with an exact 1:1 token budget (56,231 tokens) from the Stage C pretraining corpus.
3. **Generalization & Modal Collapse Finding**:
   - Synthetic pretraining dramatically reduced template loss (from 6.36 to **0.0016**) and achieved **100.0% EOS completion**.
   - Answer-token cross-entropy dropped from 10.08 to **2.12**.
   - However, greedy generation on the held-out test set **collapsed entirely to the constant digit `6`** (100% of outputs).
   - The recorded 2/10 (20%) held-out exact match was an artifact of the modal attractor matching test examples whose true sum happened to be 6 (`0 + 6` and `1 + 5`).
   - True held-out mathematical reasoning remains at 0%.

---

## Benchmark Metrics Across Conditions

| Metric | Baseline (Step 10k) | Condition A (Control) | Condition B (Arithmetic Pretraining) | Overfit Control (3 exs) |
|:---|:---:|:---:|:---:|:---:|
| **Train Exact Match** | 0 / 30 (0.0%) | 0 / 30 (0.0%) | 3 / 30 (10.0%)* | 5 / 30 (16.7%) |
| **Validation Exact Match** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 2 / 10 (20.0%)* | 0 / 10 (0.0%) |
| **Held-Out Test Exact Match** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 2 / 10 (20.0%)* | 3 / 10 (30.0%)* |
| **Symmetry Diagnostic** | 0 / 11 (0.0%) | 0 / 11 (0.0%) | 2 / 11 (18.2%)* | 1 / 11 (9.1%)* |
| **Mean Train Answer Loss** | 10.0813 | 11.2043 | **2.3183** | 9.2810 |
| **Mean Val Answer Loss** | 10.0336 | 11.0931 | **2.0942** | 10.0986 |
| **Mean Test Answer Loss** | 10.2717 | 11.1422 | **2.1251** | 7.7603 |
| **Mean Template Loss** | 6.3623 | 6.8029 | **0.0016** | 0.2135 |
| **EOS Completion Rate** | 0.0% | 0.0% | **100.0%** | **100.0%** |
| **Modal Attractor** | Wikitable Loops | Wikitable Loops | **`6` (100% of responses)** | `6` (77%), `7` (23%) |

*\*All non-zero exact matches in Condition B are artifacts of predicting the constant digit `6`.*

---

## Hardware Execution Environment

- **Architecture**: Intel x86_64, Windows 11.
- **CPU**: Intel Core i5-1135G7 @ 2.40GHz (4 cores, 8 logical processors).
- **RAM**: 31.64 GB Total System RAM.
- **PyTorch Version**: 2.14.0+cpu.
- **GPU / CUDA**: Not available (`torch.cuda.is_available() == False`). Execution was strictly CPU-bounded.
