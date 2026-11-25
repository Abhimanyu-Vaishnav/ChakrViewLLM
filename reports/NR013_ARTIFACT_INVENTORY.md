# CHAKRVIEW RESEARCH REPORT: NR-013 ARTIFACT INVENTORY
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  

---

## 1. Newly Created Datasets & Generators
1. `data/coding_sft_013/generate_coding_dataset.py`: Python script constructing and sandboxed-verifying 87 coding samples with 100% subprocess assertion pass rate.
2. `data/coding_sft_013/coding_train.jsonl`: 69 verified Python training samples.
3. `data/coding_sft_013/coding_val.jsonl`: 9 verified Python validation samples.
4. `data/coding_sft_013/coding_test.jsonl`: 9 sealed Python test samples.
5. `data/stage_e_corpus_013/build_stage_e.py`: Stage-E composite builder combining Stage-D (4,686 train, 551 val) and weighted coding dataset (552 train).
6. `data/stage_e_corpus_013/stage_e_train.jsonl`: 5,238 balanced samples.
7. `data/stage_e_corpus_013/stage_e_val.jsonl`: 559 validation samples.

---

## 2. Sealed Hardened Benchmark Suite
1. `data/sealed_eval_013/sealed_benchmark_210.py`: Script defining the 210-task hardened benchmark.
2. `data/sealed_eval_013/sealed_benchmark_210.json`: 210 fresh unseen benchmark tasks across 10 categories.
3. `experiments/baseline_eval_013/run_baseline_eval.py`: Independent sandboxed evaluation harness (subprocess execution with 2.0s timeout).
4. `experiments/baseline_eval_013/baseline_eval_summary.json`: Summary baseline evaluation log.
5. `experiments/baseline_eval_013/baseline_eval_details.json`: Turn-by-turn detailed baseline log.

---

## 3. Training Scripts & Checkpoints
1. `experiments/genesis_stage_e_013/train_17m_stage_e.py`: Fine-tuning script with corrected autoregressive shift alignment and AMP FP16.
2. `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt`: `00c15185085ab8ab2d729a5fb5ed81434412eaf57e5e6b8633fc864edda9f02c` (68,292,561 bytes).
3. `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_final.pt`: `fbddb8c75b3c754e1cf2f27622e92e5525b84f104110db048947bb89f2d2d36b` (68,292,659 bytes).
4. `experiments/stage_e_eval_013/run_stage_e_eval.py`: Candidate evaluation runner.
5. `experiments/stage_e_eval_013/stage_e_eval_summary.json`: Summary Stage-E evaluation log.
6. `experiments/stage_e_eval_013/stage_e_eval_details.json`: Turn-by-turn detailed Stage-E evaluation log.

---

## 4. Complete Research Documentation Suite
1. `reports/NR013_RECOVERY_AUDIT.md`
2. `reports/NR013_NR012_FORENSIC_AUDIT.md`
3. `reports/NR013_CODING_DATA_DESIGN.md`
4. `reports/NR013_CODING_DATA_QUALITY.md`
5. `reports/NR013_BASELINE_EVALUATION.md`
6. `reports/NR013_TRAINING_REPORT.md`
7. `reports/NR013_CAPABILITY_EVALUATION.md`
8. `reports/NR013_COGNITIVE_EVALUATION.md`
9. `reports/NR013_REGRESSION_REPORT.md`
10. `reports/NR013_ARTIFACT_INVENTORY.md`
11. `reports/NR013_FINAL_REPORT.md`
