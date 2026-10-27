# Research Evidence Dossier: Experiment 19 (Exp19)
## Debiased SFT & Template Concentration Investigation

- **Archive Status**: DIAGNOSTIC_INSIGHT_INTEGRATED
- **Artifact Status**: LOCAL_ONLY (Diagnostic Reports & Datasets)
- **Primary Evidence Directory**: experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/ and exp19_diagnostic_20261024_120300/

---

### 1. Purpose and Hypothesis
- **Purpose**: Investigate whether repetitive template structures in SFT data create artificial loss plateaus and token repetition loops during generation.
- **Hypothesis**: Debiasing template frequency and introducing lexical variation will reduce repetitive copy loops and improve held-out response diversity.

### 2. Configuration and Training Setup
- **Architecture**: 3.44M parameter ChakrMicro Transformer.
- **Training**: 600 CPU optimization steps with template concentration monitoring.
- **Diagnostics**: Loss-mask audit (LOSS_MASK_AUDIT.md) and token frequency analysis.

### 3. Empirical Results & Findings
- **Outcome**: Template concentration was significantly reduced; however, response length collapsed when template guidance was removed without explicit slot-filling constraints.
- **Distinction**: Separated in-distribution template memorization from true linguistic generation.

### 4. Supporting Artifacts & Availability
- Final Report: experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/FINAL_REPORT.md
- Diagnostic Audit: experiments/overnight_runs/exp19_diagnostic_20261024_120300/EXP19_DIAGNOSTIC_REPORT.md
- Run Manifest: experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/RUN_MANIFEST.json
