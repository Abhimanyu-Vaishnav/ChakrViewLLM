# CR-EXP-07 SCIENTIFIC CLAIM REASSESSMENT MATRIX
## Independent Post-Experiment Rigor & Evidence Classification

This document presents a granular, evidence-based classification of every major claim made in the **CR-EXP-07 Final Scientific Report** (`FINAL_SCIENTIFIC_REPORT.md`), assessing each claim against empirical data extracted in `intervention_effectiveness_audit.json`, `information_flow_metric_audit.json`, and `output_prior_audit.json`.

---

### Claim Classification Standard

- **VERIFIED**: Substantiated by direct mathematical identity, exact cryptographic validation, or replicated deterministic measurements with zero discrepancy.
- **SUPPORTED**: Directly supported by empirical measurements, but subject to specific observational limitations or scope constraints.
- **PLAUSIBLE**: Mechanistically consistent with the observations, but not conclusively isolated from alternative confounding mechanisms.
- **UNRESOLVED**: Conflicting or insufficient data prevents definitive confirmation or falsification.
- **UNSUPPORTED**: Directly contradicted by the empirical data or invalidated due to methodology/nomenclature defects.

---

### Granular Claim Matrix

| Claim ID | Claim Description | Original Report Status | Post-Review Audit Status | Empirical Justification / Corrective Finding |
|---|---|---|---|---|
| **CLM-01** | **Preflight & Checkpoint Identity Verification**<br>All 4 checkpoints, canonical baseline, vocab, and runtime session are bit-exact and unmodified ($\Delta W = 0$). | Claimed | **VERIFIED** | Checkpoint hashes match bit-exact: Parent (`327118ce...`), Pilot (`ea6ee570...`), Control Arm (`729d5812...`), Treatment Arm (`2d8cea93...`). Session hash is `9080ff51...`. Re-verified before and after all hooks. |
| **CLM-02** | **Subject & Value Intermediate Encoding**<br>Model representations at Layer 5 encode subject identity (15.4% divergence) and value identity (7.5% divergence). | Claimed | **SUPPORTED** | Verified that feeding `"Delhi"` vs `"Jaipur"` creates an $L_2$ delta of 2.1843 on a mean norm of 15.5044 at Layer 5 (14.09% relative divergence). Subject swap creates an $L_2$ delta of 15.43%. Representations do differentiate entities. |
| **CLM-03** | **Difference-in-Differences Subject-Value Interaction**<br>Internal representations form a coupled subject-value interaction vector with norm 0.3494 (2.29% of mean representation norm). | Claimed | **VERIFIED** | The $2 \times 2$ factorial counterfactual tensor $(R_A - R_B) - (A_A - A_B)$ strictly produces a non-zero residual norm of 0.3494, confirming mathematical non-linearity and contextual coupling inside the residual stream. |
| **CLM-04** | **Counterfactual Reassignment Fails Completely at Generation**<br>Model scores 0/5 (0.0%) on fact reassignments, emitting identical conversational filler `"Please result me know..."`. | Claimed | **VERIFIED** | Confirmed across all 5 reassignment prompts. Greedy generation emits polite filler tokens with 0% entity output. |
| **CLM-05** | **Signal-to-Noise Ratio (SNR) Metrics (10.4% at L0, 14.1% at L5, 7.5% at Logits)** | Claimed as "SNR" | **UNSUPPORTED** (Nomenclature) / **VERIFIED** (Arithmetic) | The mathematical formula used is $\frac{\|x_A - x_B\|_2}{0.5(\|x_A\|_2 + \|x_B\|_2)} \times 100\%$. This is **Relative Vector Divergence (Normalized Euclidean L2 Delta)**, *not* Signal-to-Noise Ratio ($\sigma_{\text{signal}}^2 / \sigma_{\text{noise}}^2$). The arithmetic values are exact, but calling it "SNR" is formally incorrect and should be corrected to "Relative Vector Divergence". |
| **CLM-06** | **Contextual Signal Survives to Answer Delimiter Position**<br>The delta between prompt facts persists through all 6 transformer layers and produces a 7.49% divergence in raw logits. | Claimed | **SUPPORTED** | Logit vector delta is 11.4154 on mean norm 152.4596 (7.49% divergence). Intermediate layer deltas do not vanish to zero. |
| **CLM-07** | **Causal Interventions Induce Logit Shifts Towards Source Fact**<br>Swapping activations at intermediate layers causally shifts downstream logits towards the donor prompt. | Claimed | **SUPPORTED** (with Position Correction) | In the original script, hardcoded slices ([2, 3] and [6, 7]) caused L5 shifts to be 0.0000. When tested at the true token positions ([12, 13, 14] for city), activation replacement produces substantial logit shift ($\Delta L_2 = 7.25 - 11.57$), doubles the target token probability ($0.000026 \to 0.000052$), and shifts target rank from 149 to 87. |
| **CLM-08** | **Causal Interventions Can Alter Output Behavior or Argmax**<br>Hypothesis that activation swapping could restore relational answers. | Refuted in report | **VERIFIED (Confirmed Negative)** | Across all 20 intervention conditions (and all corrected position conditions), the top-1 argmax was **NEVER** altered. Top-1 remained Token ID 35 (`' '`) with $P > 0.78$. Target entity rank remained $>85$. |
| **CLM-09** | **Output-Head Prioritization Failure Mechanism**<br>The generation failure is caused by an overpowering static conversational opener prior ($P > 0.73$ on Token 35 `' '`), which suppresses contextual information. | Claimed | **VERIFIED** | Audited across all 4 checkpoints. Token ID 35 receives $P = 0.8615$ in Pilot, $P = 0.7934$ in Control Arm, and $P = 0.7749$ in Treatment Arm. The target entity token (`"Delhi"`, Token 71) has $P = 0.000069$ (rank 73). The logit gap between prior and target is $>9.3$ natural log units ($e^{9.3} \approx 10,900\times$). |
| **CLM-10** | **Training Objective Misalignment as Root Cause**<br>The failure originates from conversational dialogue SFT conditioning the model to emit generic conversational turn-openers at delimiters. | Claimed | **PLAUSIBLE** | SFT dialogue data predominantly begins assistant turns with spaces, greetings, or polite acknowledgments. This matches the exact observed token distribution at the `"Answer:"` delimiter, although lack of pre-training scale is a co-factor. |

---

### Audit Summary

1. **Empirical Grounding**: The core factual observations of CR-EXP-07 are **robustly verified**. The contextual signal does reach Layer 5 and the pre-softmax logit layer, and the model's generation failure is definitively proven to be caused by the massive probability assigned to Token 35 (`' '`).
2. **Nomenclature Rectification**: The metric reported as "Signal-to-Noise Ratio (SNR)" must be formally relabeled as **Relative Vector Divergence**.
3. **Intervention Specificity**: Correcting the entity token slices demonstrates an even stronger causal coupling between intermediate activations and target token probabilities than originally reported, further reinforcing that the bottleneck is localized to the output head logit distribution.

