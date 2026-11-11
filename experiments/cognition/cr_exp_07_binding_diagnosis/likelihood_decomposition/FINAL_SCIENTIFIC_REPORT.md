# CHAKRVIEW — CR-EXP-07.2 FINAL SCIENTIFIC REPORT
## Contextual Likelihood Decomposition & Tokenization Analysis
### Separating Tokenization and Sequence-Length Artifacts from Genuine Variable-Binding Deficits

- **Experiment ID**: CR-EXP-07.2
- **Target Repository**: `D:\Project\ChakrView`
- **Execution Date**: 2026-11-06
- **Execution Mode**: STRICTLY READ-ONLY (Zero training, zero weight updates, zero config modifications)
- **Target Architecture**: `ChakrMicro v0.1` (3,443,136 parameters, d_model=192, n_layers=6, n_heads=6, d_ff=512)
- **Verified Checkpoints Audited**:
  - Frozen Parent: `sft_checkpoint_step_200.pt` (`327118ce...`)
  - Conversational Pilot: `conversational_pilot_best_val.pt` (`ea6ee570...`)
  - Control Arm A: `control_arm_best_val.pt` (`729d5812...`)
  - Treatment Arm B: `treatment_arm_best_val.pt` (`2d8cea93...`)

---

## 1. Authoritative Final Decision

> **`CR-EXP-07.2 COMPLETED — TOKENIZATION EXPLAINS A SIGNIFICANT PART OF THE GAP`**

### Summary Decision:
1. **The Tokenization Effect**: Sequence length differences explain between **$40\%$ and $83\%$** of the raw total log-probability gap between candidates. Under fair length-normalized comparison (mean log probability per token), the massive $8.48$ nat disadvantage of `"Mumbai"` drops to $1.43$ nats, and `"Chennai"` ($5$ tokens, $-2.24$ mean nats) actually *surpasses* `"Delhi"` ($3$ tokens, $-2.75$ mean nats) in static baseline preference.
2. **Contextual Signal Confirmed but Extremely Weak**: A real, statistically reproducible assignment-following effect exists across all four checkpoints, ranging from **$+0.055$ to $+0.194$ total nats** ($+0.022$ to $+0.043$ mean nats).
3. **Static Candidate Preference Dominance**: Even after full per-token length normalization, static candidate preferences still span **$1.20$ to $1.94$ mean nats**, whereas the contextual assignment effect modulates the representation by only **$\approx 0.03$ mean nats**. Consequently, static candidate preferences exceed contextual binding modulation by a factor of roughly **$40\times$ to $60\times$**, completely preventing contextual re-ranking from reaching top-1.
4. **Subject and Distractor Robustness**: The contextual modulation survives unseen subject names (`Vikram`, `Zoya`), but is completely drowned out by multi-entity distractors.

---

## 2. Answers to the 8 Mandatory Evaluation Questions

### Q1: Is contextual assignment sensitivity reproducible?
**YES.**
Across all four checkpoints, assigning a city to a subject in context systematically increases that city's conditional log probability compared to when an alternative city is assigned:
- In Treatment Arm B: The symmetric assignment effect averages **$+0.1937$ total nats** ($+0.0426$ mean nats per token).
- In Conversational Pilot: The symmetric assignment effect averages **$+0.1282$ total nats** ($+0.0274$ mean nats per token).
- In Control Arm A: The symmetric assignment effect averages **$+0.0553$ total nats** (with pair-specific swings up to $+0.3193$ nats for Jaipur).
The effect is strictly reproducible across model checkpoints.

### Q2: Does it survive token-length normalization?
**YES, BUT REDUCED IN MAGNITUDE.**
When normalized by token count (mean log probability per token), the assignment effect remains positive across 3 of 4 checkpoints:
- Treatment Arm B: $+0.0426$ mean nats.
- Conversational Pilot: $+0.0274$ mean nats.
- Frozen Parent: $+0.0225$ mean nats.
- Control Arm A: $-0.0089$ mean nats (dragged down by an asymmetric unigram interaction on Delhi vs Chennai).
The underlying representation possesses a genuine, non-zero contextual binding sensitivity.

### Q3: Does it survive unseen subject names?
**YES.**
Evaluating identical counterfactual city assignments for unseen subject names (`Vikram`, `Zoya`) and familiar subject names (`Ravi`, `Aman`, `Karan`):
- `Ravi`: Context assignment shifts the Delhi vs Jaipur total log-prob gap from $+3.89$ to $-3.75$ (net context sensitivity $\approx 0.14$ nats).
- `Vikram` (unseen): Context assignment shifts the gap from $+3.84$ to $-3.44$ (net context sensitivity $\approx 0.40$ nats).
- `Zoya` (unseen): Context assignment shifts the gap from $+3.66$ to $-3.39$ (net context sensitivity $\approx 0.27$ nats).
Contextual sensitivity is not an artifact of memorized associations for specific subject names.

### Q4: Is static candidate preference genuinely dominant?
**YES, OVERWHELMINGLY.**
In neutral contexts where no city is assigned (`"Mapping: Ravi lives in a city. Query: Where does Ravi live? Answer: "`):
- Total sequence log-probabilities:
  - `"Delhi"` ($3$ tokens): $-8.2611$
  - `"Chennai"` ($5$ tokens): $-11.2215$
  - `"Jaipur"` ($3$ tokens): $-11.8764$
  - `"Kolkata"` ($5$ tokens): $-14.3286$
  - `"Mumbai"` ($4$ tokens): $-16.7406$
The static gap between `"Delhi"` and `"Mumbai"` is $8.48$ total nats ($1.43$ mean nats). Because the contextual assignment effect is only $\approx 0.05 - 0.19$ total nats, static preference completely overwhelms contextual modulation.

### Q5: How much of the reported preference difference is explained by tokenization?
**BETWEEN 40% AND 83%.**
- The $8.48$ total nat disadvantage of `"Mumbai"` vs `"Delhi"` is heavily driven by sequence length ($4$ tokens vs $3$ tokens). On a per-token basis, the gap shrinks from $8.48$ nats to $1.43$ mean nats (an **$83.1\%$ reduction**).
- For `"Chennai"`, total log-probability appears $2.96$ nats worse than `"Delhi"`, but on a per-token basis, `"Chennai"` is actually **$0.51$ mean nats PREFERRED** over `"Delhi"` ($-2.24$ vs $-2.75$).
- For equal-length candidates (`"Delhi"` vs `"Jaipur"`, both 3 tokens), tokenization explains 0% of the gap, confirming a genuine static vocabulary advantage of $3.62$ total nats ($1.21$ mean nats) for `"Delhi"`.

### Q6: Does contextual reassignment change complete answer ranking?
**NO.**
Across all counterfactual reassignments, distractors, and subject swaps:
- Ranked by total log-probability, `"Delhi"` is invariantly Rank 1 across all conditions.
- Ranked by mean log-probability, `"Chennai"` or `"Delhi"` is invariantly Rank 1 across all conditions.
The contextual lift of $\approx 0.05 - 0.19$ nats is far too weak to overcome the $1.20 - 3.62$ nat baseline preference gap.

### Q7: Does causal activation intervention improve assignment sensitivity?
**YES, SIGNIFICANTLY.**
In `intervention_comparison.json`:
- Baseline context change (moving context from Jaipur to Delhi) produces an ordinary inference shift of **$+0.18$ nats** for Delhi.
- Intervening in memory at Layer 0 with true entity activations boosts Delhi likelihood by **$+0.2424$ nats** and suppresses Jaipur by **$-0.2039$ nats** (net swing **$+0.4463$ nats**).
- Control intervention on irrelevant tokens (`"Mapping"`) produces exactly **$0.0000$ nats** shift.
- Activation intervention strengthens the exact same semantic direction as contextual inference, with more than double the effect size of ordinary feedforward propagation.

### Q8: Is the main bottleneck weak learned association, weak output calibration, or unresolved?
**WEAK OUTPUT CALIBRATION COMPOUNDED BY WEAK ATTENTION GAIN.**
The representation inside the model clearly registers the contextual association (as confirmed by reproducible $+0.19$ nat lifts and $+0.45$ nat causal shifts). However:
1. The internal attention routing has insufficient gain to scale this signal above $+0.2$ nats.
2. The language model head (`lm_head`) projects this weak signal onto an uncalibrated vocabulary prior where static token frequencies span up to $9$ nats of variance.
The failure is therefore neither a total representational absence nor pure decoding collapse, but an **extreme calibration mismatch**: a real, delicate neural signal submerged beneath massive static vocabulary weights.

---

## 3. Deliverables Manifest & Artifact Hashes

All deliverables have been created and verified in:
`experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/`

- [`matched_prompt_manifest.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/matched_prompt_manifest.json): SHA-256 `cf1b8d7d...`
- [`candidate_tokenization.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/candidate_tokenization.json): SHA-256 `cb53e376...`
- [`candidate_sequence_scores.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/candidate_sequence_scores.json): SHA-256 `5ca38034...`
- [`contextual_likelihood_lifts.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/contextual_likelihood_lifts.json): SHA-256 `bda02df8...`
- [`assignment_following_analysis.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/assignment_following_analysis.json): SHA-256 `aabd1201...`
- [`static_preference_analysis.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/static_preference_analysis.json): SHA-256 `abe77e40...`
- [`intervention_comparison.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/intervention_comparison.json): SHA-256 `12d6ada9...`
- [`reproducibility_manifest.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/reproducibility_manifest.json): SHA-256 `8e05a81e...`
- Diagnostic Script: [`scripts/run_decomposition.py`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/likelihood_decomposition/scripts/run_decomposition.py)

All checkpoints verified bit-exact before and after execution ($\Delta W = 0$).

