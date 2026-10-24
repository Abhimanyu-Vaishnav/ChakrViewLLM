# MORNING HANDOFF — CHAKRVIEW v0.8 OVERNIGHT AUTONOMOUS MISSION

**Mission Directory**: `experiments/overnight_runs/mission_v08_overnight_20261023_232924/`  
**Execution Timestamp**: 2026-10-23T23:29:24+05:30 to 2026-10-23T23:46:00+05:30  
**Environment**: Windows 11, Intel Core (14 cores / 20 threads), 31.64 GB RAM, PyTorch 2.14.0+cpu  
**Git Branch**: `dev/v0.6-v0.7-validation` | **HEAD Commit**: `f30136a5dc02934180a503fdf0a4c287469bf106`  
**Final Release Gate Verdict**: **`CONDITIONAL_HOLD`**

---

## 1. Executive Summary & Questions Answered

### What did you actually inspect?
- Hardware environment, processes, memory, and disk space.
- Git repository status and HEAD commit.
- All 8 protected repository assets and their canonical SHA-256 hashes.
- `chakrview/evaluation/task_evaluator.py` and its Factual QA heuristics.
- Exp18 generator candidate banks and unit tests in `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/`.
- Training engine `chakrview/training/sft_engine.py` and prompt contract `chakrview/training/prompt_contract.py`.
- Neural runtime adapter `chakrview/runtime/neural_runtime_adapter.py`.
- The repository test suite in `tests/`.

### What did you change, and why?
- **[task_evaluator.py](file:///d:/Project/ChakrView/chakrview/evaluation/task_evaluator.py)**: Fixed `_score_factual_qa` by removing legacy hardcoded heuristics that mapped any question with "desert" to "antarctic" and any question with "particle" to "neutron". The updated logic dynamically extracts target entities matching the ground-truth answer (e.g. Sahara desert, electron, proton, neutron).
- **[test_task_evaluator.py](file:///d:/Project/ChakrView/tests/test_task_evaluator.py)**: Added regression test `test_factual_qa_varied_entities` covering Sahara, electron, proton, and mismatch detection.
- **Mission Artifacts**: All mission logs, datasets, checkpoints, and reports were created strictly inside `experiments/overnight_runs/mission_v08_overnight_20261023_232924/`.

### Which bugs were reproduced and fixed?
- **Reproduced**: Running diagnostic `debug_fqa.py` on the Exp18 candidate bank revealed 4 false negatives in `_score_factual_qa` (`EXP18_TRAIN_FACTUAL_QA_0031`, `0124`, `EXP18_VAL_FACTUAL_QA_0006`, `0007`).
- **Fixed**: Generalized focal entity matching. Re-evaluating the 1,520 records on expected responses produced **0 false negatives** (1,330 / 1,330 scorable records scored 100% CORRECT).

### Which tests were run, and what were the actual results?
1. `tests/test_task_evaluator.py`: **17 / 17 PASSED** (Exit Code 0).
2. `test_exp18_generator.py`: **15 / 15 PASSED** (Exit Code 0).
3. Core unittest discover suite: **123 / 123 PASSED** in 25.9s (Exit Code 0).
4. Stage E training pipeline sanity: **PASSED** (tied parameters verified: 3,443,136; causal masking verified: prompt masked with -100; finite loss and optimizer step verified).

### Did Exp18 generation and audit pass independently?
- **Yes**: Generated all 1,520 records in memory (1,200 train, 160 val, 160 test) across 8 categories (190 each).
- **Split Isolation**: Exactly 0 cross-split prompt leaks. 15 shared responses were verified as standard refusal templates in the uncertainty/abstention category.
- **Serialization**: No official JSONL files were written to the Exp18 repository folder, strictly adhering to the freeze rule.

### Were all factual and semantic quality concerns resolved, or are some still pending?
- 190 short-explanation items are held out in [SEMANTIC_REVIEW_QUEUE.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/SEMANTIC_REVIEW_QUEUE.csv) with status `NEEDS_REVIEW`. They require human expert scientific domain review.

### Was training actually executed? How many optimizer steps completed?
- **Yes**: Executed 120 optimizer steps on CPU using `SFTExperimentEngine`.
- **Source Checkpoint**: Isolated copy of canonical `step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Data Used**: Isolated Exp18 training split (1,200 records) and validation split (160 records). The 160 test records were strictly held out.
- **Best Experimental Result**: `sft_best_checkpoint.pt` achieved at Step 80 (validation loss: 3.3562, down from initial 4.6159).

### Did baseline-versus-experiment evaluation show a genuine improvement?
- **Loss & Termination**: Training improved cross-entropy loss from 4.6159 to 3.3562 and increased EOS completion rate from 0.0% to 8.75% (+8.75%).
- **Task Correctness**: On the 140 scorable held-out test records, baseline achieved 2.14% (3/140) while the 120-step SFT model achieved 1.43% (2/140) (delta: -0.71%).
- **Structured JSON Validity**: Remained at 0.0% on both checkpoints (45-token generation horizon under greedy decoding does not close valid JSON syntax).
- **Conclusion**: The experimental checkpoint shows improved language termination (EOS) and perplexity, but does NOT demonstrate task accuracy improvement.

### Did any protected hash change?
- **NO**: All 8 protected hashes match their authoritative references 100% bitwise:
  - Canonical Checkpoint: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**MATCH**)
  - Vocab `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (**MATCH**)
  - Merges `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (**MATCH**)
  - Runtime `session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (**MATCH**)
  - Model `model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (**MATCH**)
  - Exp17 Train: `074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181` (**MATCH**)
  - Exp17 Val: `15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564` (**MATCH**)
  - Exp17 Test: `be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430` (**MATCH**)

---

## 2. Benchmark & Evaluation Metrics

| Metric | Baseline (`step_10000.pt`) | SFT Best (`step_80.pt`) | Delta | Evaluator Denominator |
| :--- | :---: | :---: | :---: | :--- |
| **Scorable Task Accuracy** | **2.14%** (3/140) | **1.43%** (2/140) | -0.71% | 140 scorable test records |
| **EOS Completion Rate** | **0.00%** (0/160) | **8.75%** (14/160) | +8.75% | 160 total test records |
| **Structured JSON Validity** | **0.00%** (0/20) | **0.00%** (0/20) | 0.00% | 20 structured formatting test records |
| **Validation Loss** | 4.6159 | 3.3562 | -1.2597 | 160 val records (sampled batches) |
| **CPU Latency per Record** | 296.91 ms | 351.27 ms | +54.36 ms | Greedy decoding, max 45 new tokens |

---

## 3. Mission Deliverables Index

All 26 required deliverables are present in `experiments/overnight_runs/mission_v08_overnight_20261023_232924/`:
1. [MORNING_HANDOFF.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/MORNING_HANDOFF.md)
2. [RELEASE_GATE_REPORT.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/RELEASE_GATE_REPORT.md)
3. [MISSION_PLAN.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/MISSION_PLAN.md)
4. [COMMAND_LOG.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/COMMAND_LOG.txt)
5. [INITIAL_GIT_STATUS.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/INITIAL_GIT_STATUS.txt)
6. [FINAL_GIT_STATUS.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/FINAL_GIT_STATUS.txt)
7. [INITIAL_PROTECTED_HASHES.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/INITIAL_PROTECTED_HASHES.json)
8. [FINAL_PROTECTED_HASHES.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/FINAL_PROTECTED_HASHES.json)
9. [ENVIRONMENT_REPORT.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/ENVIRONMENT_REPORT.json)
10. [PROCESS_AND_RESOURCE_REPORT.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/PROCESS_AND_RESOURCE_REPORT.json)
11. [TEST_EXECUTION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/TEST_EXECUTION.json)
12. [RAW_TEST_OUTPUT.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/RAW_TEST_OUTPUT.txt)
13. [DATASET_AUDIT_SUMMARY.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/DATASET_AUDIT_SUMMARY.json)
14. [CATEGORY_SPLIT_COUNTS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/CATEGORY_SPLIT_COUNTS.csv)
15. [LEAKAGE_FINDINGS.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/LEAKAGE_FINDINGS.csv)
16. [SEMANTIC_REVIEW_QUEUE.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/SEMANTIC_REVIEW_QUEUE.csv)
17. [EVALUATOR_AUDIT.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/EVALUATOR_AUDIT.json)
18. [BASELINE_EVALUATION.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/BASELINE_EVALUATION.json)
19. [TRAINING_CONFIG.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/TRAINING_CONFIG.json)
20. [TRAINING_LOG.csv](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/TRAINING_LOG.csv)
21. [EXPERIMENT_COMPARISON.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/EXPERIMENT_COMPARISON.json)
22. [INTEGRITY_MANIFEST.json](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/INTEGRITY_MANIFEST.json)
23. [FINAL_DATASET_GATE.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/FINAL_DATASET_GATE.md)
24. [FINAL_TRAINING_GATE.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/FINAL_TRAINING_GATE.md)
25. [V08_RELEASE_CHECKLIST.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/V08_RELEASE_CHECKLIST.md)
26. [KNOWN_ISSUES.md](file:///d:/Project/ChakrView/experiments/overnight_runs/mission_v08_overnight_20261023_232924/KNOWN_ISSUES.md)

---

## 4. Exact Reproduction Command for Human Reviewers

To reproduce the primary verification steps in a single command, run:
```powershell
& "d:\Project\ChakrView\.venv\Scripts\python.exe" "experiments/overnight_runs/mission_v08_overnight_20261023_232924/stage_j_final_integrity.py"
& "d:\Project\ChakrView\.venv\Scripts\python.exe" -m unittest -v tests/test_task_evaluator.py
& "d:\Project\ChakrView\.venv\Scripts\python.exe" -m unittest -v experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/tests/test_exp18_generator.py
```
