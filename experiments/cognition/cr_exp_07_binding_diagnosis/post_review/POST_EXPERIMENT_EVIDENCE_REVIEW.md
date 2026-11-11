# CHAKRVIEW — CR-EXP-07 POST-EXPERIMENT EVIDENCE REVIEW
## Rigorous Read-Only Audit of Contextual Information Flow and Generation Bottlenecks

- **Review Type**: Post-Experiment Evidence Review & Mechanistic Audit
- **Target Experiment**: CR-EXP-07 (*Subject Identity & Context Binding Diagnosis*)
- **Target Artifacts Audited**:
  - `causal_intervention_results.json`
  - `information_flow_results.json`
  - `subject_value_interaction_results.json`
  - `counterfactual_reassignment_results.json`
  - `FINAL_SCIENTIFIC_REPORT.md`
- **Execution Mode**: STRICTLY READ-ONLY (Zero training, zero weight updates, zero config modifications)
- **Primary Checkpoints Audited**:
  - Control Arm A (`control_arm_best_val.pt` — `729d5812...`)
  - Treatment Arm B (`treatment_arm_best_val.pt` — `2d8cea93...`)
  - Conversational Pilot (`conversational_pilot_best_val.pt` — `ea6ee570...`)
  - Frozen Parent (`sft_checkpoint_step_200.pt` — `327118ce...`)

---

## 1. Executive Summary & Core Verdict

The reported core conclusion of CR-EXP-07 is:
> *"Contextual relational information reaches the answer position through the transformer layers, but is completely overridden at generation by an overpowering static conversational prior."*

### Audit Decision: **SUBSTANTIATED WITH NOMENCLATURE CORRECTION**

The empirical evidence rigorously supports the finding that contextual entity information persists through all 6 transformer layers into the pre-softmax logits, but cannot be decoded into generated text because of an extreme output-head probability prior ($P > 0.77$) on Token ID 35 (`' '`). 

However, this review identifies two critical methodological findings:
1. **Nomenclature Rectification**: The metric reported as "Signal-to-Noise Ratio (SNR %)" is mathematically **Relative Vector Divergence** ($\frac{\|x_A - x_B\|}{0.5(\|x_A\| + \|x_B\|)}$), not an information-theoretic variance ratio.
2. **Causal Intervention Efficacy**: In the original experiment, hardcoded token slices misaligned with entity boundaries at Layer 5. When evaluated at exact BPE token boundaries, causal activation intervention produces an even stronger logit shift ($\Delta L_2 = 7.25 - 11.57$) and doubles the target entity token probability, while strictly confirming that linear activation replacement alone is insufficient to escape the $>9.3$ logit unit deficit against the static opener prior.

---

## 2. Answers to the 8 Mandatory Evaluation Questions

### Q1: Does the evidence substantiate that contextual information reaches the answer position?
**YES.**
Across both Frozen Parent, Conversational Pilot, and CR-EXP-06 Checkpoints, altering the context entity (e.g., swapping `"Delhi"` for `"Jaipur"`) produces a non-zero, persistent divergence across every layer:
- At Layer 0: Residual stream difference $\Delta = 0.5085$ (10.43% relative divergence).
- At Layer 5: Residual stream difference $\Delta = 2.1843$ (14.09% relative divergence).
- At Final Logits: Logit difference $\Delta = 11.4154$ (7.49% relative divergence).
The contextual delta does not decay to zero, confirming that information survives forward propagation to the final delimiter token position.

### Q2: What is the exact mathematical definition of the reported 10.4%, 14.1%, and 7.5% metrics?
The metric was implemented as:
$$\text{Metric} = \frac{\|x_A - x_B\|_2}{\frac{1}{2}(\|x_A\|_2 + \|x_B\|_2)} \times 100\%$$
- **Layer 0**: $\frac{0.5085}{4.8743} \times 100\% = 10.43\%$
- **Layer 5**: $\frac{2.1843}{15.5044} \times 100\% = 14.09\%$
- **Final Logits**: $\frac{11.4154}{152.4596} \times 100\% = 7.49\%$
**Mathematical Assessment**: This is **Relative Vector Divergence (Normalized Euclidean Distance)**. It does not calculate signal power relative to noise variance ($\sigma_s^2 / \sigma_n^2$) and must be formally designated as Relative Divergence rather than SNR.

### Q3: What is the exact token identity and probability distribution of the reported conversational prior?
The static conversational prior is dominated by **Token ID 35** (ASCII space `' '`):
- **Conversational Pilot**: Token 35 receives **$86.15\%$** probability (Logit: $11.5854$).
- **Control Arm A**: Token 35 receives **$79.34\%$** probability (Logit: $11.3957$).
- **Treatment Arm B**: Token 35 receives **$77.49\%$** probability (Logit: $11.0719$).
- **Frozen Parent Baseline**: Token 898 (`'The '`) receives $28.99\%$ probability, while Token 35 receives $1.42\%$.
In contrast, the correct contextual entity token (Token 71, `'D'` in `"Delhi"`) receives a probability of only **$0.000069$** ($0.0069\%$, rank 73, logit 2.0509) in Control Arm A. The logit gap is $11.3957 - 2.0509 = 9.3448$ units ($e^{9.3448} \approx 11,439\times$).

### Q4: Did causal activation interventions alter top-1 predictions or sequence likelihood?
**NO.**
Across all 20 causal interventions in `causal_intervention_results.json` and all re-tested exact-slice interventions, the top-1 prediction was **never altered**. It remained locked to Token 35 with $P > 0.77$. 
However, intervening with source value activations at true token slices ([12, 13, 14]) doubled the target token's probability ($0.000026 \to 0.000052$) and improved its rank from 149 to 87, confirming direct causal transmission that is nonetheless too small to change the argmax.

### Q5: How specific were the causal interventions compared to control perturbations?
**HIGHLY SPECIFIC.**
- Interventions on entity tokens produced $L_2$ logit shifts of **$7.25$ to $15.08$**.
- Control perturbations at irrelevant prefix tokens (`"Mapping:"`) produced $L_2$ logit shifts of **$0.00$ to $0.08$**.
The causal impact of activation replacement is strictly localized to entity token positions.

### Q6: Does the Difference-in-Differences interaction prove relational binding or merely additive superposition?
**PARTIAL (WEAK COUPLING).**
The $2 \times 2$ factorial counterfactual test yielded an interaction vector norm of **$0.3494$**, representing **$2.29\%$** of the mean representation norm. Because this norm is non-zero, the representation is strictly not purely linear superposition. However, an interaction magnitude of only $2.29\%$ indicates very weak non-linear binding, explaining why downstream attention heads fail to route the bound entity to the output.

### Q7: What is the primary bottleneck preventing relational extraction?
The failure is an **output-head prioritization bottleneck (lm_head projection misalignment)** coupled with **weak attention binding**. The SFT dialogue tuning conditioned the output head weights to heavily prioritize conversational turn-openers (spaces, polite conversational tokens) upon encountering question delimiters, while the internal transformer layers lack sufficient relational binding energy to override this prior.

### Q8: What are the actionable recommendations for CR-EXP-08?
1. **Targeted Logit Bias Recalibration / Contrastive Decoding**: Test whether suppressing the top conversational prior tokens at inference time reveals the preserved contextual entity signal.
2. **Relational Delimiter Conditioning**: Adapt the prompt structure or SFT masking to prevent assistant delimiters from triggering dialogue opener priors.
3. **Loss-Masked Relational SFT**: If training is authorized in the future, supervise only the final target answer token while freezing or regularizing the base representations.

---

## 3. Audit Artifact Manifest

All generated evidence files are preserved in:
`experiments/cognition/cr_exp_07_binding_diagnosis/post_review/`

- `POST_EXPERIMENT_EVIDENCE_REVIEW.md`: This comprehensive master evaluation report.
- `intervention_effectiveness_audit.json`: Detailed logit shifts, target ranks, and probabilities across all interventions.
- `information_flow_metric_audit.json`: Mathematical audit of layerwise divergences, numerators, and denominators.
- `output_prior_audit.json`: Cross-checkpoint evaluation of Token 35 and target entity probabilities.
- `claim_reassessment.md`: Granular classification of all CR-EXP-07 claims into Verified, Supported, Plausible, or Unsupported.
- `reproducibility_manifest.json`: Cryptographic SHA-256 manifest of all relevant project artifacts.

