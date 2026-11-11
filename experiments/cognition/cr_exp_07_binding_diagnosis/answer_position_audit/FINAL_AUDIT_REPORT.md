# CHAKRVIEW — CR-EXP-07.1 FINAL AUDIT REPORT
## Answer-Position & Token-Conditional Probability Audit
### Disentangling Whitespace Syntax from Semantic Output Priors

- **Experiment ID**: CR-EXP-07.1
- **Target Repository**: `D:\Project\ChakrView`
- **Audit Execution Date**: 2026-11-06
- **Execution Mode**: Strictly Read-Only Audit (Zero parameter updates, zero training, zero architecture modifications)
- **Target Architecture**: `ChakrMicro v0.1` (3,443,136 parameters, d_model=192, n_layers=6, n_heads=6, d_ff=512)
- **Verified Checkpoints Audited**:
  - Frozen Parent: `sft_checkpoint_step_200.pt` (`327118ce...`)
  - Conversational Pilot: `conversational_pilot_best_val.pt` (`ea6ee570...`)
  - Control Arm A: `control_arm_best_val.pt` (`729d5812...`)
  - Treatment Arm B: `treatment_arm_best_val.pt` (`2d8cea93...`)

---

## 1. Authoritative Final Decision

> **`CR-EXP-07.1 COMPLETED — MIXED OUTPUT BOTTLENECK`**

### Summary Decision:
1. **The Whitespace Finding**: Token ID 35 (`' '`) dominating the next-token distribution after `Answer:` with $P = 77.5\% - 86.2\%$ is **syntactically expected leading whitespace**, not an epistemic conversational prior. Evaluating semantic entity probabilities directly at `Answer:` was a token-position misalignment confounder.
2. **The True Semantic Output Bottleneck**: Supplying the expected whitespace prefix (`Answer: `) does **not** restore entity generation. The model predicts newline (`'\n'`, $P \approx 25\%$) and comma (`', '`, $P \approx 10-16\%$), and greedy decoding emits conversational noise (`"\n\nShandard the context.\n\nN"`).
3. **Log-Likelihood Disconnect**: Teacher-forced candidate evaluation across candidate entities (`Delhi`, `Jaipur`, `Mumbai`, `Chennai`, `Kolkata`) demonstrates that static token frequency biases differ by **$3.8$ to $9.0$ nats**, completely swamping the genuine but weak contextual binding signal (**$\approx 0.32$ nats**).
4. **Causal Interventions**: Swapping true entity activations shifts candidate log-likelihoods by $+0.15$ to $+0.45$ nats specifically toward the donor entity (with $0.0000$ shift on control tokens), proving direct mechanistic transmission that is nonetheless an order of magnitude too small to overcome the static prior.

---

## 2. Granular Stage-by-Stage Results

### 2.1 Stage 1: Tokenizer Ground Truth (`TOKENIZATION_GROUND_TRUTH.json`)
- In `vocab_4096`, `Answer:` encodes to 6 tokens: `[68, 113, 118, 122, 283, 61]` (`['A', 'n', 's', 'w', 'er', ':']`).
- When followed by a space, `Answer: ` merges the last two tokens into merged token `2550` (`'er: '`), producing 5 tokens: `[68, 113, 118, 122, 2550]`.
- Entity representations after `Answer: `:
  - `"Delhi"`: `[71, 904, 453]` (`['D', 'el', 'hi']`) — no leading space token needed because token 2550 already contains the trailing space.
  - `"Jaipur"`: `[77, 1000, 2382]` (`['J', 'ai', 'pur']`)
  - `"Chennai"`: `[70, 107, 275, 709, 108]` (`['C', 'h', 'en', 'na', 'i']`)
  - `"Mumbai"`: `[80, 120, 550, 1000]` (`['M', 'u', 'mb', 'ai']`)
  - `"Kolkata"`: `[78, 1142, 110, 278, 100]` (`['K', 'ol', 'k', 'at', 'a']`)
- All strings roundtrip identically through `decode(encode(text)) == text`.

### 2.2 Stage 2: Correct Answer-Position Evaluation (`ANSWER_POSITION_DISTRIBUTIONS.json`)
Evaluating the next-token distribution under both prompt terminations:
- **Condition A (`Answer:`)**:
  - Parent: Top-1 = Token 898 (`'The '`, $28.99\%$), Token 35 (`' '`, $1.42\%$)
  - Pilot: Top-1 = Token 35 (`' '`, $86.15\%$)
  - Control Arm: Top-1 = Token 35 (`' '`, $79.34\%$)
  - Treatment Arm: Top-1 = Token 35 (`' '`, $77.49\%$)
  - *Interpretation*: The conversational fine-tuning taught the model standard English orthography (space after colon).
- **Condition B (`Answer: ` — correctly tokenized with trailing space)**:
  - Parent: Top-1 = Token 96 (`']'`, $6.99\%$), Top-2 = Token 81 (`'N'`, $5.74\%$), Top-3 = Token 898 (`'The '`, $5.11\%$)
  - Pilot: Top-1 = Token 13 (`'\n'`, $25.54\%$), Top-2 = Token 276 (`', '`, $9.96\%$), Top-3 = Token 96 (`']'`, $4.94\%$)
  - Control Arm: Top-1 = Token 13 (`'\n'`, $24.44\%$), Top-2 = Token 276 (`', '`, $16.27\%$), Top-3 = Token 898 (`'The '`, $4.45\%$)
  - Treatment Arm: Top-1 = Token 13 (`'\n'`, $25.79\%$), Top-2 = Token 276 (`', '`, $15.35\%$), Top-3 = Token 898 (`'The '`, $5.04\%$)
  - *Interpretation*: When whitespace is provided, the model predicts newlines and conversational punctuation, still completely failing to produce the target entity.

### 2.3 Stage 3 & 4: Conditional Sequence Likelihood & Counterfactual Reassignment (`CONDITIONAL_ANSWER_LIKELIHOODS.json`, `COUNTERFACTUAL_SEQUENCE_EVALUATION.json`)
Evaluating candidate sequence teacher-forced log probabilities under `Answer: ` across counterfactual assignments:
1. **Static Unigram Advantage**:
   - Total Sequence Log Probabilities across candidates (Control Arm, Ravi in Jaipur):
     - `"Delhi"`: $-7.9578$ (Mean LP: $-2.6526$)
     - `"Chennai"`: $-10.9822$ (Mean LP: $-2.1964$)
     - `"Jaipur"`: $-11.7089$ (Mean LP: $-3.9030$)
     - `"Kolkata"`: $-14.1873$ (Mean LP: $-2.8375$)
     - `"Mumbai"`: $-16.9659$ (Mean LP: $-4.2415$)
   - `"Delhi"` has a static unigram advantage of **$+3.75$ nats** over `"Jaipur"` and **$+9.01$ nats** over `"Mumbai"`.
2. **Contextual Modulation Magnitude**:
   - When context assigns `Ravi -> Delhi`: Total LP for Delhi is $-8.1405$.
   - When context assigns `Ravi -> Jaipur`: Total LP for Delhi is $-7.9578$, while Total LP for Jaipur increases from $-12.0282$ to $-11.7089$ ($\Delta = +0.3193$ nats).
   - The model has a genuine context sensitivity of **$\approx 0.32$ nats**, but because the static vocabulary bias favoring Delhi is **$3.75$ nats**, the candidate ranking is invariant.

### 2.4 Stage 5: Reassessing Activation Interventions (`INTERVENTION_REASSESSMENT.json`)
Evaluating in-memory activation swaps at true entity slices ([12, 13, 14]) on full candidate sequence likelihoods:
- Swapping Delhi activations into a Jaipur prompt at Layer 0 increases Delhi sequence log probability by **$+0.2424$ nats** and suppresses Jaipur by **$-0.2039$ nats** (Net swing: **$+0.4463$ nats** toward Delhi).
- At Layer 2: Net swing = **$+0.3441$ nats** toward Delhi.
- At Layer 4: Net swing = **$+0.1533$ nats** toward Delhi.
- Control intervention at irrelevant tokens (`"Mapping"`, tokens 0–3): Net swing = **$0.0000$ nats**.
- **Conclusion**: Causal activation interventions measurably alter semantic sequence log-likelihoods, confirming genuine semantic information transmission. However, the $\approx 0.45$ nat shift cannot bridge the $3.75$ nat static vocabulary prior gap.

---

## 3. Comparative Summary Across Deliverables

| Deliverable Artifact | Key Empirical Verification |
|---|---|
| [`TOKENIZATION_GROUND_TRUTH.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/TOKENIZATION_GROUND_TRUTH.json) | Proves `Answer:` ends in `:` (needs whitespace), whereas `Answer: ` merges to `er: ` (ready for entity). |
| [`ANSWER_POSITION_DISTRIBUTIONS.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/ANSWER_POSITION_DISTRIBUTIONS.json) | Shows Token 35 drops from $>77\%$ at `Answer:` to $<0.01\%$ at `Answer: `, where `'\n'` takes over at $25\%$. |
| [`CONDITIONAL_ANSWER_LIKELIHOODS.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/CONDITIONAL_ANSWER_LIKELIHOODS.json) | Full teacher-forced log probs and token-by-token ranks across Delhi, Jaipur, Chennai, Mumbai, and Kolkata. |
| [`COUNTERFACTUAL_SEQUENCE_EVALUATION.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/COUNTERFACTUAL_SEQUENCE_EVALUATION.json) | Proves contextual modulation is small ($\approx 0.32$ nats) relative to unigram priors ($3.8 - 9.0$ nats). |
| [`INTERVENTION_REASSESSMENT.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/INTERVENTION_REASSESSMENT.json) | Demonstrates causal activation swaps induce $+0.45$ nat shift toward donor city on full sequences. |
| [`OUTPUT_PRIOR_REASSESSMENT.md`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/OUTPUT_PRIOR_REASSESSMENT.md) | Refutes Token 35 as semantic prior; verifies mixed syntactic + semantic output distribution bottleneck. |
| [`REPRODUCIBILITY_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/cognition/cr_exp_07_binding_diagnosis/answer_position_audit/REPRODUCIBILITY_MANIFEST.json) | Bit-exact cryptographic SHA-256 validation for all models, tokenizer, and audit outputs ($\Delta W = 0$). |

---

## 4. Final Scientific Synthesis & Recommendations

1. **Clarification of CR-EXP-07**: CR-EXP-07 was correct that contextual information reaches the final layer and that generation fails at the output head. However, identifying Token 35 (`' '`) as an "epistemic conversational prior" was an artifact of prompt-ending tokenization. Token 35 is simply expected whitespace.
2. **The True Nature of the Bottleneck**: The true failure is a **Mixed Output Bottleneck**:
   - **Syntactic**: When whitespace is missing, the model emits whitespace.
   - **Semantic Prior**: When whitespace is present, static vocabulary unigram priors and conversational continuation patterns (newlines, commas) overpower the subtle contextual binding signal ($0.32$ nats of contextual modulation vs $3.8 - 9.0$ nats of static prior).
3. **Actionable Pathway**:
   - **Inference-Time**: Contrastive decoding or prior-subtraction decoding (evaluating $P(\text{cand} | \text{prompt}) / P(\text{cand} | \text{null\_prompt})$) can immediately amplify the $0.32$ nat contextual signal over the static vocabulary prior.
   - **Training-Time**: Relational fine-tuning must be supervised specifically on the semantic token position (following the delimiter space) with cross-entropy loss applied solely to target entities to scale up contextual binding energy.

