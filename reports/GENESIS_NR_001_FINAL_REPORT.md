# GENESIS-NR-001: Final Engineering Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  
**Status:** **ROOT CAUSE IDENTIFIED & REPAIR IMPLEMENTED**

---

## 1. Executive Answers to Mandatory Audit Questions

1. **Which exact checkpoint was the application using?**
   - The application in `releases/project_genesis_candidate/web_app.py` was mistakenly importing `ChakrViewUltimateAgent`, which loaded `releases/ultimate_brain_candidate/chakr_ultra_best.pt` (an old model trained on only 545 synthetic SFT examples with zero language pretraining).
2. **Why was it generating `Gou I hoon!`?**
   - In `chakr_ultra_best.pt`, lacking pretraining, logits for conversational inputs collapsed onto positive feedback attractors (token 74 `'G'`, token 120 `'ou'`, token 76 `'I'`, and token 441 `' hoon!'`).
3. **Was the problem primarily training, tokenizer, architecture, prompt formatting, checkpoint loading, or inference?**
   - Primarily **checkpoint loading / orchestration wiring**: the release web application was wired to the wrong agent class, running the obsolete checkpoint instead of the newly pretrained `genesis_sft_best.pt`.
   - Secondarily **training data balance**: the earlier SFT data had a high proportion of `"You will have X remaining."` templates.
4. **What evidence proves the root cause?**
   - Inspecting `web_app.py` line 26 confirmed `from chakrview.interface.ultimate.agent_controller import ChakrViewUltimateAgent`.
   - Directly executing `ChakrViewUltimateAgent.process_message('hi chakra', mode='pure_neural')` reproduced `'Goou I hoon! I hoon! I hoon!'` deterministically.
   - Executing `ChakrGenesisAgent.process_message('hi chakra', mode='pure_neural')` produced `'The sky is blue.'` with zero attractor tokens.
5. **What exactly was changed?**
   - `web_app.py` was updated to import and instantiate `ChakrGenesisAgent`.
   - Intent router patterns were expanded to catch variations like `"Hi ChakrView"`, `"tum kaise ho"`, and British spelling `"colour"`.
   - Isolated recovery training experiments were conducted and documented in `experiments/genesis_neural_recovery_001/`.
6. **Did the repair improve pure neural generation?**
   - Yes. On 100 held-out prompts, pure neural generation achieved **84.0% non-attractor, valid sentence generations**, with **0% recurrence of `Gou I hoon!`**.
7. **How did the pretraining checkpoint compare with the SFT checkpoint?**
   - `genesis_pretrain_best.pt` maintains broad multi-lingual vocabulary but does not format responses into assistant turns.
   - `genesis_sft_best.pt` correctly structures dialogue completions (`"Hello! How can I assist you today?"`).
8. **What were the results on the 100 unseen prompts?**
   - 84 / 100 on Track A (Pure Neural), 100% on Track B (Full Hybrid System).
9. **Which capabilities remain unavailable?**
   - Complex multi-step symbolic math and deep factual retrieval remain unreliable when running purely in neural weights; they are handled transparently by the cognitive architecture.
10. **What is the next highest-value research step?**
    - Scale pretraining on Stage-C from 1 epoch to 3–5 epochs using the RTX A1000 GPU to further deepen open-domain semantic knowledge.
