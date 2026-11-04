# CBR-SFT-02 — DEEP FORENSIC FAILURE ANALYSIS & SCIENTIFIC ROOT CAUSE

---

## 1. The Core Scientific Question

In CBR-SFT-02:
1. The 2-example controlled overfit test proved that the neural core **can** learn and output conversational dialogue (`"The secret code is 999."` and `"I am ChakrView."`) with 0.0004 loss.
2. The training loss on the 540-example mixture decreased from 5.23 to 4.11, and validation perplexity dropped from 185 to 61.5.
3. Attractor collapse was halved from 53.1% to 28.1%.
4. **Yet on free generation across 60 M3 tasks, 100 entity tasks, and 50 multi-turn tasks, accuracy remained 0.0%.**

Why did conversational accuracy remain 0%?

---

## 2. Root Cause Breakdown

### A. Pretraining Prior Dominance vs. Gradient Budget
- **The Scale Mismatch:** The model was pretrained for 10,000 steps on 6.65 million tokens of synthetic Stage C corpus text. The attention weights are heavily conditioned on Stage C continuation attractors.
- In CBR-SFT-02, fine-tuning for 200 steps at $LR=5\times 10^{-5}$ on 540 examples (~150,000 tokens) provides enough gradient magnitude to suppress runaway loops, but **not enough gradient magnitude to overcome the deeply entrenched pretraining prior** across 540 distinct topics.
- In the 2-example test, 50 steps at $LR=10^{-3}$ was sufficient to overwrite the prior for those 2 prompts. For 540 diverse prompts, 200 steps at $5\times 10^{-5}$ merely scratches the surface of the loss landscape.

### B. Single-Step Free Generation Divergence
- In teacher forcing, the model receives ground-truth prompt and response prefixes, yielding low loss.
- In free autoregressive generation, if the very first token emitted after `Assistant:` is not the exact intended word (e.g. emitting `"The"` instead of `"Good"`), the model immediately falls back into its dominant pretraining attractor basin (`"The first of the word of the world..."`).

### C. The Absence of Pretrained Inductive Copy Circuits
- As established in M5.1, the base ChakrMicro weights lack functional induction heads for copying arbitrary dynamic entities from context.
- Supervised fine-tuning on diverse text cannot create attention induction circuits from scratch in 200 steps.

---

## 3. What Did Work
- Chat scaffolding eliminated Wikipedia table markup emissions entirely.
- Native intelligence improved to 3/32 (9.4%).
- Attractor collapse rate dropped from 53.1% to 28.1%.
- Python coding syntax validity reached 33.3%.
