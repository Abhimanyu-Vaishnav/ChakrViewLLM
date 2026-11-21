# GENESIS-NR-001: Repair Log

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  

---

## 1. Summary of Repairs Executed

1. **Repaired `releases/project_genesis_candidate/web_app.py`:**
   - Switched import from `ChakrViewUltimateAgent` to `ChakrGenesisAgent`.
   - Updated initialization to load `ChakrGenesisAgent()` directly, binding the web interface to `genesis_sft_best.pt` on the GPU.
2. **Updated `chakrview/reasoning/ultimate/intent_router.py` & Release Candidate:**
   - Expanded regex patterns to recognize conversational greetings (`"Hi ChakrView"`, `"tum kaise ho"`), spelling variants of colour in abstention queries, and natural user corrections (`"That was incorrect. It is Canberra."`).
3. **Created Isolated Training Experiment in `experiments/genesis_neural_recovery_001/`:**
   - `build_balanced_dataset.py`: Balanced SFT dataset with 464 non-repetitive conversational pairs.
   - `train_recovery.py`: Isolated recovery trainer producing `genesis_recovery_sft_best.pt`.
   - `run_100_evaluation.py`: 100-prompt unseen evaluation suite.
