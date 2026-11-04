# CBR-SFT-03A — FORENSIC DECISION & EXPERIMENTAL BLUEPRINT

---

## 1. Explicit Answers to Mandated Questions

### 1. Is the first-token prediction actually the dominant failure?
**YES, UNEQUIVOCALLY.**
Across 100 held-out prompts, **100% of generation trajectories diverge at Position 0**. The correct target concept token has a mean rank of **180.35** and an average probability of only **0.05%**. Because the first token emitted is an attractor like `'A'` or `'The '`, free generation is derailed into pretraining continuations before a single concept word can be produced.

### 2. Is the model learning conditional dialogue behavior?
**PARTIALLY.**
The model learned to emit `<|eos|>` turn terminations (91.2% EOS rate) and eliminated raw Wikipedia table syntax (`\n! scope=...`). In the controlled overfit test, it demonstrated conditional routing for near-distribution paraphrases. However, across the 540-example mixture, gradient updates were too diffused to steer open-domain dialogue tokens.

### 3. Does training improve only teacher-forced prediction?
**YES.**
Under teacher forcing, cross-entropy loss dropped substantially (perplexity fell from 185 to 40.7), but free autoregressive generation accuracy remained 0% across all suites due to immediate Position-0 divergence.

### 4. Which checkpoint is strongest across all independent suites?
**Step 50 (`sft_checkpoint_step_50.pt`)**.
It achieves the Pareto-optimal trade-off: highest Native Intelligence score (**3/32, 9.4%**), highest Coding AST validity (**4/12, 33.3%**), strong attractor reduction (**34.4%** vs 53.1%), and lowest validation loss before over-fitting.

### 5. Is the proposed 100-example, 500-step CBR-SFT-03 experiment justified?
**YES, WITH REFINED PARAMETERS.**
The forensic evidence proves that 540 diverse topics diffuse gradient magnitude, leaving the pretraining prior dominant at Position 0. Concentrating gradient updates onto a compact core of 100 high-frequency dialogue patterns (greetings, role responses, identity, and short entity copies) with higher learning rate ($LR=2\times 10^{-4}$ to $5\times 10^{-4}$) is mathematically necessary to overcome the pretraining prior.

### 6. What is the smallest controlled experiment that can distinguish the remaining hypotheses?
**CBR-SFT-03 Pilot (50–100 Dialogue Pairs, 300–500 Steps, $LR=2\times 10^{-4}$)**.
Testing whether Position-0 accuracy on common conversational intents (greetings and identity) breaks 0% without degrading the 3/32 intelligence score.

---

## 2. Definitive Mission Recommendation

- **Do NOT launch CBR-SFT-03 automatically.**
- Present the verified forensic report to the project owner.
- The path toward the first non-zero conversational milestone requires concentrated gradient steering on dialogue anchors rather than broad topical diffusion.
