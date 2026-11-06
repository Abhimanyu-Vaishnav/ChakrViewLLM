# CHAKRVIEW — CONVERSATIONAL CANDIDATE EVALUATION REPORT (STAGE 5)

## 1. Candidate Identity & Training Lineage
- **Candidate Checkpoint**: `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt`
- **SHA-256**: `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd`
- **Parent Lineage**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` (`327118ce...`)
- **Architecture**: Frozen ChakrMicro v0.1 ($3,443,136$ parameters, $L=6, d=192, H=6, d_{\text{ff}}=512$, tied embeddings)
- **Training Config**: 100 optimizer steps, effective batch size 16 (micro-batch 4, accumulation 4), AdamW ($\text{lr}=5\times 10^{-5} \to 1\times 10^{-5}$ cosine decay, $\text{weight\_decay}=0.05$), gradient clip $1.0$, loss computed exclusively on assistant response tokens.

---

## 2. Key Empirical Findings
1. **Zero Output Mode Collapse**:
   - The primary finding from CR-EXP-01 Stage A was catastrophic mode collapse into `'Cube'` / `'Mumbai'` due to unregularized batch-size 1 updates.
   - The new candidate completely resolves this pathology: zero instances of `'Cube'` or infinite table tokens occurred across any of the 18 evaluation prompts or 54 held-out test samples.
2. **Held-out Loss Reduction**:
   - On the 54 held-out test samples (unseen during training or validation), test cross-entropy loss dropped from **$5.2353$** (Parent) to **$3.0480$** (Candidate), representing an impressive **$41.78\%$ relative reduction in perplexity**.
3. **Epistemic Abstention Acquisition**:
   - When asked for missing variables or unknown information, the candidate accurately outputs:
     `"That information is not provided in the context."`
     (Confirmed on Test Sample 4 and Prompt D1).
4. **Current Limitations & Boundary of Capability**:
   - While the candidate has learned conversational flow, greetings, persona adherence, and epistemic abstention, exact factual key-value binding in long, multi-turn contexts ($T > 100$) remains noisy. The model occasionally slips into code docstring fragments (`\n    >>> link_top...`) when prompt context grows complex.

---

## 3. Evaluation Artifacts
- Evaluation Data: `experiments/cognition/autonomous_12h/conversational_evaluation_results.json`
- Capability Registry: `experiments/cognition/autonomous_12h/CAPABILITY_REGISTRY.md`

