# ChakrView Master Sprint Batches 83–86: Final Report

## Executive Summary

Sprint Batches 83–86 executed two parallel tracks:
1. **Track A (Research)**: Evaluated whether explicit intermediate counting scratchpads (`Start at a; count b more: ...; answer: c`) enable ChakrMicro v0.1 (3.44M parameters) to overcome first-operand heuristic collapse and generalize to unseen addition problems.
2. **Track B (Product)**: Built and verified the **ChakrView Local Workbench**, an interactive zero-dependency web dashboard and CLI for inspecting model checkpoints, running local inferences, comparing outputs against deterministic oracles, and viewing historical reports.

---

## 1. Track A: Scratchpad Learning Experiment Findings

- **Training Setup**: Matched 1:1 token budget (99,417 tokens) trained for 100 steps from canonical step-10,000 checkpoint comparing direct arithmetic (Condition A) vs scratchpad arithmetic (Condition B).
- **Format Adherence**: Condition B achieved **100.0% valid format rate** and **100.0% EOS completion**, generating fluent, grammatically flawless scratchpad strings.
- **Held-Out Generalization**: On genuinely held-out test operand pairs, exact-match accuracy was **0 / 18 (0.0%)**.
- **Mechanistic Cause**: Instead of dynamically counting from input $a$ by $b$ steps, the model memorized two canned scratchpad prototypes from training (`Start at 0; count 4 more: 1, 2, 3, 4; answer: 4` for small inputs and `Start at 9; count 4 more: 10, 11, 12, 13, 14, 15; answer: 15` for large inputs).
- **Scientific Takeaway**: Intermediate sequence positions alone do not guarantee algorithmic composition in small transformers; the optimization landscape still favors memorizing extended template prototypes over dynamic routing.

---

## 2. Track B: Local Workbench Deliverable

- **Architecture**: Pure Python standard library (zero external dependencies like Streamlit or npm needed).
- **Capabilities**:
  * Checkpoint verification (SHA-256 calculation, parameter count validation, size inspection).
  * Local CPU inference engine using the frozen 4096-token BPE tokenizer.
  * Deterministic Arithmetic Oracle comparator with explicit provenance tags (`DETERMINISTIC_ORACLE` vs `CHAKRMICRO_LOCAL_MODEL`).
  * Report persistence and viewing across historical sprint benchmarks.
- **Launch Command**:
  ```powershell
  .\.venv\Scripts\python.exe experiments/batch83_86_scratchpad_workbench/workbench/app.py serve --port 8501
  ```
- **CLI Commands**: `inspect`, `infer`, `oracle`, `serve`.
- **Smoke Test**: Passed end-to-end unit and CLI verification.

---

## 3. Quantitative Summary Table

| Condition | Format | Train Match | Val Match | Test Match | Format Validity | EOS Rate | Loss | Output Pattern |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Step 10k)** | Direct | 0 / 64 (0%) | 0 / 18 (0%) | 0 / 18 (0%) | 0.0% | 0.0% | 6.7330 | Wikitable loops |
| **Baseline (Step 10k)** | Scratchpad | 0 / 64 (0%) | 0 / 18 (0%) | 0 / 18 (0%) | 0.0% | 0.0% | 5.6089 | Wikitable loops |
| **Condition A (Direct)** | Direct | 6 / 64 (9.4%) | 0 / 18 (0%) | 3 / 18 (16.7%)* | 100.0% | 100.0% | 0.2567 | `8` (100% modal) |
| **Condition B (Scratchpad)** | Scratchpad | 3 / 64 (4.7%) | 2 / 18 (11.1%) | **0 / 18 (0.0%)** | **100.0%** | **100.0%** | **0.3081** | Canned scratchpads (`4` & `15`) |

*\*Note: 3/18 in Condition A was an artifact of modal digit 8 matching test items where sum = 8.*

---

## 4. Checkpoints and Verification

- Model Parameters: **3,443,136**
- Fresh Init SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Base Step-10k Checkpoint SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Condition A Checkpoint: [`condition_a_direct_arithmetic_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch83_86_scratchpad_workbench/experiments/condition_a_direct_arithmetic_step_100.pt) (SHA-256: `cc554926216a40cd51d438874352595ae940e7f2ee105a5239ed6bea01cc2766`)
- Condition B Checkpoint: [`condition_b_scratchpad_arithmetic_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch83_86_scratchpad_workbench/experiments/condition_b_scratchpad_arithmetic_step_100.pt) (SHA-256: `f55b87aff8b2c4d7e46f2336fc39e4929c171497bde0c6508c48a823fd8e318d`)
- User File Preserved: [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) remains untouched and uncommitted.
- Regression Suite: **115 passing tests** across 21 test files (100% pass rate).
