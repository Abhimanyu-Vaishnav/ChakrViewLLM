# CHAKRVIEW RESEARCH REPORT: NR-014 ARTIFACT INVENTORY
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  

---

## 1. Datasets & Builders
1. `data/coding_sft_014/build_coding_dataset.py`: Parametric code generator producing 195 verified coding samples.
2. `data/coding_sft_014/expand_dataset.py`: Expanded builder producing 234 verified coding tasks across geometry, finance, algorithms, and units.
3. `data/coding_sft_014/coding_train_expanded.jsonl`: 234 verified Python coding training samples.
4. `data/stage_f_corpus_014/build_stage_f.py`: Balanced composite corpus builder (5,388 train, 570 val).
5. `data/stage_f_corpus_014/stage_f_train.jsonl`: Stage-F training dataset.
6. `data/stage_f_corpus_014/stage_f_val.jsonl`: Stage-F validation dataset.

---

## 2. Hardened Generalization Benchmark (260 Tasks)
1. `data/sealed_eval_014/build_benchmark_260.py`: Benchmark creation script.
2. `data/sealed_eval_014/sealed_benchmark_260.json`: 260 sealed tasks across 10 categories.
3. `experiments/eval_260_014/run_eval_260.py`: Sandboxed subprocess evaluation harness.
4. `experiments/eval_260_014/eval_260_summary.json`: Summary comparison across 4 configurations.
5. `experiments/eval_260_014/eval_260_details.json`: Full turn-by-turn evaluation records.

---

## 3. Test Suites, Training Scripts & Checkpoints
1. `tests/test_training_objective_nr014.py`: Automated causal LM objective verification suite.
2. `experiments/genesis_stage_f_014/train_17m_stage_f.py`: Stage-F balanced training pipeline.
3. `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt`: `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` (68,292,561 bytes).
4. `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_final.pt`: `0a5eb83f804c043ca5f167da4255a4e74080c2e764298e856368627a0c13d96b` (68,292,659 bytes).

---

## 4. Formal Research Reports
1. `reports/NR014_RECOVERY_AUDIT.md`
2. `reports/NR014_TRAINING_OBJECTIVE_AUDIT.md`
3. `reports/NR014_STAGE_E_FAILURE_ANALYSIS.md`
4. `reports/NR014_CODING_DATA_DESIGN.md`
5. `reports/NR014_CODING_DATA_QUALITY.md`
6. `reports/NR014_BASELINE_EVALUATION.md`
7. `reports/NR014_TRAINING_REPORT.md`
8. `reports/NR014_GENERALIZATION_REPORT.md`
9. `reports/NR014_REGRESSION_REPORT.md`
10. `reports/NR014_ARTIFACT_INVENTORY.md`
11. `reports/NR014_FINAL_REPORT.md`
