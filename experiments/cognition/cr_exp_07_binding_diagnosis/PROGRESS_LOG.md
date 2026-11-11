# CR-EXP-07 Autonomous Investigation Progress Log

- **Experiment ID**: CR-EXP-07 Subject Identity & Context Binding Diagnosis
- **Investigation Type**: Neural representation and causal information-flow investigation
- **Start Timestamp**: 2026-11-06T22:18:00+05:30
- **Execution Limit**: Maximum 12 hours (Read-Only)
- **Git Branch**: `exp/cbr-01-conversational-breakthrough`
- **Git HEAD**: `564472a42245a47a7bf6f78ca6410e523383d2ce`
- **Host Specs**: Windows 11 AMD64, 20 logical CPUs, 31.64 GB RAM (15.65 GB available), Python 3.14.7, PyTorch 2.14.0+cpu

---

## Stage Progress Tracker

| Stage | Description | Status | Completion Timestamp | Key Artifacts / Deliverables |
|---|---|---|---|---|
| **Stage 0** | Checkpoint & Implementation Preflight | COMPLETED | 2026-11-06T22:19:55+05:30 | `REPOSITORY_SAFETY_AUDIT.md`, `checkpoint_identity_audit.json` |
| **Stage 1** | Tokenization & Controlled Input Design | COMPLETED | 2026-11-06T22:20:29+05:30 | `tokenization_analysis.json` |
| **Stage 2** | Subject Sensitivity Diagnostics | COMPLETED | 2026-11-06T22:21:31+05:30 | `subject_sensitivity_results.json` |
| **Stage 3** | Value Sensitivity Diagnostics | COMPLETED | 2026-11-06T22:21:31+05:30 | `value_sensitivity_results.json` |
| **Stage 4** | Subject-Value Interaction (2x2 Counterfactual) | COMPLETED | 2026-11-06T22:22:41+05:30 | `subject_value_interaction_results.json` |
| **Stage 5** | Counterfactual Reassignment Evaluation | COMPLETED | 2026-11-06T22:23:15+05:30 | `counterfactual_reassignment_results.json` |
| **Stage 6** | Causal Activation Interventions (A, B, C, D) | COMPLETED | 2026-11-06T22:23:46+05:30 | `causal_intervention_results.json` |
| **Stage 7** | Multi-Layer Information Flow Diagnostics | COMPLETED | 2026-11-06T22:24:14+05:30 | `information_flow_results.json` |
| **Stage 8** | Robustness and Generalization Suite | COMPLETED | 2026-11-06T22:26:13+05:30 | `robustness_results.json` |
| **Stage 9** | Scientific Synthesis & Final Deliverables | COMPLETED | 2026-11-06T22:27:46+05:30 | `claim_evidence_matrix.md`, `UPDATED_CAPABILITY_REGISTRY.md`, `reproducibility_manifest.json`, `FINAL_SCIENTIFIC_REPORT.md` |

---

## Execution Journal
- **22:18:00**: Initialized directory structure, logged system hardware parameters, and created progress journal.
- **22:19:55**: Stage 0 verified all 4 model checkpoints, canonical baseline, tokenizer artifacts, and runtime files.
- **22:20:29**: Stage 1 established controlled input designs and entity tokenization asymmetries.
- **22:21:31**: Stages 2 & 3 confirmed intermediate subject sensitivity (15.4% divergence) and value sensitivity (7.5% divergence).
- **22:22:41**: Stage 4 measured non-zero 2x2 counterfactual interaction tensor (norm=0.3494, 2.29%).
- **22:23:15**: Stage 5 demonstrated 0% counterfactual reassignment capability during autoregressive generation.
- **22:23:46**: Stage 6 executed in-memory causal activation replacements; confirmed parameter integrity verified bit-exact ($\Delta W = 0$).
- **22:24:14**: Stage 7 traced signal-to-noise across layers; identified Mechanism 4 & 5 (signal survives to Layer 5 but is overridden by static output prior).
- **22:26:13**: Stage 8 completed evaluation across 450 out-of-distribution prompts (0% exact match).
- **22:27:46**: Stage 9 compiled capability matrix, manifest, and final scientific report. Autonomous investigation concluded.

