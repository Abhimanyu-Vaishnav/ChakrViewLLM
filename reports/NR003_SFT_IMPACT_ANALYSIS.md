# NR003: SFT Impact Analysis

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  

---

## 1. Capabilities Improved by SFT
- **Assistant Turn Formatting:** Transitioned the model from unformatted token continuation into structured `Assistant: <reply>` generation.
- **Greeting & Identity Adherence:** Successfully memorized full greetings, polite conversational openings, and system identity responses.
- **Loss Reduction:** Reduced loss on structured dialogue from 8.05 down to 3.52.

---

## 2. Pathologies Introduced by SFT
- **Template Memorization Sinks:** Over-exposure to repeated responses (such as `"You will have X remaining."` and `"The ocean waves crash gently..."`) turned these sequences into strong attractor baselines for out-of-distribution prompts.
- **Lack of Generalization:** Because foundational pretraining was compromised by the token permutation mismatch, SFT did not fine-tune an existing understanding of language; it merely acted as a shallow pattern memorizer.
- **Independence from Input:** When tested on novel questions (such as `"Why does a computer need RAM?"`), the model lacked conceptual grounding and emitted whatever high-frequency response cluster had the highest prior probability.
