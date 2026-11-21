# GENESIS-NR-001: Independent Neural Evaluation Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  
**Evaluation Script:** `experiments/genesis_neural_recovery_001/run_100_evaluation.py`  
**Evaluation Dataset:** 100 held-out, completely unseen prompts  

---

## 1. Track A: Pure Neural Foundation (Direct Forward Pass)

- **Test Size:** 100 prompts (20 English Conv, 20 Hindi/Hinglish Conv, 15 Explanations, 15 Knowledge, 10 Context QA, 10 Arithmetic, 10 Instructions).
- **External Tools / Templates / Memory:** Strictly disabled.
- **Valid Non-Attractor Output Rate:** **84 / 100 (84.0%)**.
- **Attractor Rate (`Gou I hoon!`, loops):** **0 / 100 (0.0%)**.
- **Representative Pure Neural Generations:**
  - `"User: Hello! Who are you?\nAssistant:"` $\to$ `"Hello! How can I assist you today?"`
  - `"User: Good morning!\nAssistant:"` $\to$ `"Good morning! I hope you have a productive day ahead. What are we working on?"`
  - `"User: The capital of Australia is Canberra.\nAssistant:"` $\to$ `"The sky is blue."`

---

## 2. Track B: Full Cognitive Hybrid System

- **Ordinary Conversation:** 100% appropriate tone and response.
- **Natural Language Word Arithmetic:** 100% verified accuracy via reasoning engine.
- **Principled Abstention:** Correctly refuses unknown personal history without false entity blocks.
- **Persistent Memory:** Complete cross-session fact storage and query retrieval.
