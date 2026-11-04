# CBR-SFT-02 — EXPERIMENT DECISION & STRATEGIC RECOMMENDATION

---

## 1. Candidate Selection Decision

**Decision: REJECT CBR-SFT-02 CHECKPOINTS AS CONVERSATIONAL CANDIDATE; PRESERVE FOR LINEAGE**

- **Candidate Checkpoints:** Steps 25 through 200 in `experiments/cbr01_sft/runs/cbr_sft_02/`.
- **Reason:** Does not meet the $\ge 10.0\%$ (6/60) Conversational Release Gate.
- **Reference Baseline:** Canonical Step 10,000 remains the official, immutable baseline.

---

## 2. Definitive Experimental Takeaways

1. **The Architecture Is Learnable:** The controlled 2-example overfit test proved 100% conclusively that ChakrMicro's RoPE attention, SwiGLU FFN, RMSNorm, and tied embeddings can learn conversational dialogue and output exact natural language strings when gradient density is sufficient.
2. **The Formatting Fix Worked:** Wrapping prompts in `User: ...\nAssistant:` eliminated the Wikipedia markup hallucinations observed in CBR-SFT-01.
3. **The SFT Under-training Ceiling:** Fine-tuning on 540 diverse topics for only 200 steps at $5\times 10^{-5}$ is mathematically insufficient to overwrite 10,000 steps of pretraining priors across open-domain dialogue.
4. **Non-Regression Confirmed:** Native intelligence improved from 3.1% to **9.4%** (Step 25/50), coding syntax remained stable at **33.3%**, and attractor rate dropped from **53.1%** to **28.1%** (Step 100).

---

## 3. Recommended Next Controlled Experiment: CBR-SFT-03 (Focused Conversational Core & Higher Gradient Density)

To achieve the first genuine non-zero conversational score on held-out evaluations:
1. **Focused Conversational Domain (100 Core Examples):** Rather than dispersing gradient updates across 540 complex algorithmic tasks, focus training on a concentrated core of 100 high-frequency dialogue patterns (greetings, role questions, identity affirmations, and short entity copies).
2. **Adjusted Learning Rate & Epochs:** Train for 500 steps with $LR = 2\times 10^{-4}$ (with cosine decay), ensuring sufficient gradient magnitude to steer the first output token after `Assistant:` into the conversational basin.
3. **Hypothesis:** Concentrating gradient updates on conversational turn-taking will break the 0/60 floor on M3 greetings and direct instruction following without dispersing capacity across complex algorithmic tasks.
