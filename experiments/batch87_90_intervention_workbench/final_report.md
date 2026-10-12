# ChakrView Master Sprint Batches 87–90: Final Report

## Executive Summary

Sprint Batches 87–90 executed two parallel tracks:
1. **Track A (Research)**: Investigated whether explicit step-by-step incremental next-token supervision ($k \to k+1$) enables ChakrMicro v0.1 (3.44M parameters) to condition each step on prior state and solve single-digit addition.
2. **Track B (Product)**: Upgraded the **ChakrView Local Workbench** to v0.2 with live asynchronous training monitoring and real-time loss plotting.

---

## 1. Track A: Incremental Supervision Experiment Findings

- **Experimental Setup**: 100 steps from canonical step-10,000 checkpoint on matched 124,289 tokens comparing direct arithmetic (Condition A) vs incremental step supervision (Condition B).
- **Direct-Answer Accuracy**:
  * Condition A (Direct): 7/64 train, 0/18 val, 2/18 test (100% collapsed to `10`).
  * Condition B (Incremental): 4/64 train, 2/18 val, **0/18 test** (100% collapsed to `13`).
- **Autonomous Multi-Step Chain Evaluation**:
  * Condition B Autonomous Chain Success on Held-Out Test: **0 / 18 (0.0%)**.
  * Transition Accuracy: **0 / 83 (0.0%)**.
- **Mechanistic Cause**: Next-token cross-entropy on incremental steps rapidly found a central attractor (`13`) minimizing global loss across all transition targets, failing to learn the successor function $S(n) = n + 1$.
- **Cold Reload Verification**: Deterministically reproduced 0/18 test exact match and 0% transition accuracy upon fresh-process reload.

---

## 2. Track B: Local Workbench v0.2 Deliverable

- **Live Job Monitoring**: Asynchronous training job launcher with real-time step progress and status reporting (`RUNNING`, `COMPLETED`, `FAILED`).
- **HTML5 Canvas Loss Curve**: Interactive real-time cross-entropy descent curves rendered directly on the browser dashboard (zero external UI dependencies).
- **Report Persistence**: Automatic saving of step-by-step loss history and metadata to `workbench/reports/`.
- **Smoke Test**: Passed all unit and CLI tests.
- **Launch Command**:
  ```powershell
  .\.venv\Scripts\python.exe experiments/batch87_90_intervention_workbench/workbench/app.py serve --port 8501
  ```

---

## 3. Verified Checkpoints and Invariants

- Canonical Architecture Parameters: **3,443,136**
- Fresh Init SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- Base Step-10k Checkpoint SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Condition A Checkpoint: [`condition_a_direct_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch87_90_intervention_workbench/experiments/condition_a_direct_step_100.pt) (SHA-256: `e97b6a46469c0b211dce484fd0f8ee1b489130ca5dc52f872c171121e3f0b739`)
- Condition B Checkpoint: [`condition_b_incremental_step_100.pt`](file:///d:/Project/ChakrView/experiments/batch87_90_intervention_workbench/experiments/condition_b_incremental_step_100.pt) (SHA-256: `f0bb3b4ffb1af0f382efe4406875bb418b6a980936e2e9295ea808d5cdc390e6`)
- Preserved User File: [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) kept untouched and uncommitted.
- Regression Test Suite: **119 passing tests** across 22 test files (100% pass rate).
