# CR-16E-R2 Git Delivery & Working Tree Report

### 1. Delivery Summary
- **Experiment:** CR-EXP-16E-R2 (Answerability Audit, Copying Recovery & Controlled Gradient-Interference Study)
- **Status:** **COMPLETE**
- **Git Branch:** `exp/cbr-01-conversational-breakthrough`
- **Output Isolation:** All outputs are strictly confined to `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study/`.
- **Pre-existing Experiment Artifacts:** Preserved read-only. None of CR-16, CR-16A, CR-16B, CR-16C, CR-16D, CR-16D-R, CR-16E, or CR-16E-R files were overwritten or modified.
- **Canonical Model & Weights:** Preserved frozen. `chakrview/runtime/session.py` verified unchanged (`9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`).
- **Git Safety:** Zero destructive Git operations performed. No unprompted commits or pushes.

---

### 2. Complete Inventory of Delivered Artifacts
Located in `experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study/`:

1. `CR16ER2_FINAL_SCIENTIFIC_REPORT.md` — Comprehensive executive scientific report.
2. `CR16ER2_PHASE_A_RECONCILIATION.md` — Parameter count ($3,443,136$) and training dataset distribution reconciliation.
3. `CR16ER2_ANSWERABILITY_AUDIT.md` — Validated answerability breakdown ($1,400$ answerable, $100$ unanswerable in CR-16C).
4. `CR16ER2_ABSTENTION_RECALCULATION.md` — Recalculated metrics, coverage, and risk-versus-coverage matrices.
5. `CR16ER2_COPYING_FORENSIC_ANALYSIS.md` — Forensic breakdown of sequence copying failure modes.
6. `CR16ER2_CONTROLLED_EXPERIMENT_DESIGN.md` — Experimental protocol for the G1 vs G2 loss-masking study.
7. `CR16ER2_G1_G2_TRAINING_REPORT.md` — Checkpoint manifests, seeds, and training logs.
8. `CR16ER2_GRADIENT_INTERFERENCE_ANALYSIS.md` — Statistical analysis proving negative loss did not cause the copying collapse.
9. `CR16ER2_SPAN_GENERALIZATION.md` — Dissection of seen vs unseen span generation ($0.0\%$ on unseen entities).
10. `CR16ER2_STATISTICAL_REPORT.md` — Paired $t$-tests and Cohen's $d$ effect sizes ($N=3$ training replications).
11. `CR16ER2_FINAL_CAPABILITY_SCORECARD.md` — 14-dimension capability registry status matrix.
12. `CR16ER2_ARCHITECTURE_DECISION.md` — Evidence-based decision freezing canonical ChakrMicro v0.1.
13. `GIT_DELIVERY_REPORT.md` — Git status and delivery manifest.
14. `ARTIFACT_HASHES.txt` — SHA-256 checksums of all 36 files.
15. `answerability_labels.json` — Machine-readable per-probe labels for all 3,300 benchmark probes.
16. `answerability_audit_summary.json` — Category-wise answerability totals.
17. `g1_g2_evaluation_master_summary.json` — Consolidated scores for all 6 trained checkpoints.
18. `checkpoints/` — 6 trained checkpoints across Arm G1 (loss enabled) and Arm G2 (loss masked).
19. `eval_outputs/` — Complete raw prediction records for all 6 models across all 6 benchmark suites.

