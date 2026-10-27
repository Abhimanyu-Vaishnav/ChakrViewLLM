# Research Archive Index: Experiments 18–30 (Exp18–Exp30)

This directory serves as the structured, permanent research archive for the ChakrView Exp18–Exp30 research campaign. It documents experimental hypotheses, training protocols, evaluation outcomes, negative results, and epistemic boundaries.

## Epistemic Integrity Principles
1. **Separation of Evidence and Inference**: Negative findings and model capability boundaries are explicitly recorded, not omitted or rationalized away.
2. **Preservation of Artifacts**: Checkpoints (.pt) and raw training traces remain in local research storage to keep Git history lightweight, while structured dossiers, summary manifests, and diagnostic tooling are versioned.
3. **Reproducibility Classification**: Every experiment states whether its results are directly reproduced, independently audited, or historically reported.

---

## Experiment Registry (Exp18–Exp30)

| ID | Title / Mechanism | Outcome & Epistemic Boundary | Status | Dossier |
| :--- | :--- | :--- | :--- | :--- |
| **Exp18** | Aligned SFT & Scientific Explanations | 20/20 human review rejections. Establishes parameter-internal reasoning limits of 3.44M core. | REJECTED_HYPOTHESIS_FORMALIZED_AS_BOUND | [EXP18.md](EXP18.md) |
| **Exp19** | Debiased SFT & Template Concentration | Mitigated template loops; demonstrated collapse in length without slot guidance. | DIAGNOSTIC_INSIGHT_INTEGRATED | [EXP19.md](EXP19.md) |
| **Exp20** | Contextual Binding & Generalization Limits | High in-distribution copy (>85%); degraded on out-of-vocabulary distractors. | EMPIRICALLY_BOUNDED | [EXP20.md](EXP20.md) |
| **Exp21** | Synthetic Recombined Copying | Improved short-span copying; unstable beyond 8 tokens. | COMMITTED_IN_GIT | [EXP21.md](EXP21.md) |
| **Exp22** | Attention Copy Loss vs Control | Auxiliary attention loss converged faster but yielded no gain over tuned cross-entropy. | COMMITTED_IN_GIT | [EXP22.md](EXP22.md) |
| **Exp23** | Single-Token Copying Diagnostic | Isolated single-token failure to softmax competition with syntactic connectors. | COMMITTED_IN_GIT | [EXP23.md](EXP23.md) |
| **Exp24** | Copying vs Routing Diagnostic | Routing demonstrated high accuracy (>90%); copying failed on out-of-vocab tokens. | COMMITTED_OR_READY_FOR_ARCHIVE | [EXP24.md](EXP24.md) |
| **Exp25** | Vocabulary Generalization Diagnostic | Quantified accuracy drop from 88.4% (seen) to 12.1% (unseen) tokens. | COMMITTED_OR_READY_FOR_ARCHIVE | [EXP25.md](EXP25.md) |
| **Exp26** | 15M Parameter Architecture Readiness | Verified latency, memory limits, and gradient scaling. Not claimed as trained capability. | COMMITTED_IN_GIT | [EXP26.md](EXP26.md) |
| **Exp27** | Pointer Forensic Audit & Gradient Diagnostics | Identified prefix leakage in prior pointer runs; diagnosed vanishing gradients. | CRITICAL_FORENSIC_AUDIT_READY_FOR_ARCHIVE | [EXP27.md](EXP27.md) |
| **Exp28** | Fair Baseline vs Pointer Comparison | Proven pointer superiority on held-out single-token copying (100% vs 0% EM); quarantined from v0.8.0 runtime due to task specificity and paraphrasing limits. | PIVOTAL_GOVERNANCE_DECISION_AUDIT | [EXP28.md](EXP28.md) |
| **Exp29** | Independent Audit & Robustness | 100% blind replication of Exp28 under distractor noise. Justified baseline freeze for v0.8.0. | INDEPENDENT_AUDIT_PASS | [EXP29.md](EXP29.md) |
| **Exp30** | Human Review Closure & Release Gate | Reconciled Exp18 human reviews; CPU runtime smoke suite passed; v0.8.0 release gate closed. | RELEASE_GATE_PASSED | [EXP30.md](EXP30.md) |

---

## Validated Diagnostic & Evaluation Tools

The following reusable tools have been reviewed, sanitized, and archived under scripts/research_tools/:
- udit_leakage_and_splits.py: Independent dataset split and leakage scanner.
- 	est_gradient_diagnostic.py: Gradient flow diagnostic harness for dual-head architectures.
-
un_exp28_fair_comparison.py: Controlled benchmark runner with randomized distractor positioning.
-
un_reproduction.py: Blind independent evaluation replication script.
-
ecompute_metrics.py: Metric recomputation engine under varying distractor distance constraints.
-
un_bounded_pilot.py: Lightweight pilot training and evaluation harness.
- uild_review_dossier.py: Automated review queue generator for blind human review.
-
econcile_human_reviews.py: Consensus reconciliation engine for multi-reviewer evaluation queues.

---

## Machine-Readable Manifest
See [archive_manifest.json](archive_manifest.json) for structured metadata mapping every experiment to its local artifact directories, audit records, and reproducible scripts.
