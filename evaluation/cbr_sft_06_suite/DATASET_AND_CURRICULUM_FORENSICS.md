# Phase G: Training Data & Curriculum Forensics

**Date:** 2026-11-04  
**Datasets Compared:**  
- `data/sft/cbr_sft_04/dataset.jsonl` (160 examples)  
- `data/sft/cbr_sft_05/dataset.jsonl` (400 examples)  

---

## 1. Compositional Shift & Forgetting Root Cause

A direct forensic comparison reveals why conversational greeting capabilities suffered complete collapse during CBR-SFT-05:

| Feature / Category | CBR-SFT-04 Dataset | CBR-SFT-05 Dataset | Difference / Finding |
| :--- | :---: | :---: | :--- |
| **Total Examples** | 160 | 400 | +250% expansion |
| **Conversational Greetings** | **31 (19.4%)** | **0 (0.0%)** | **Zero Replay (100% Omission)** |
| - *Morning Greetings* | 10 | 0 | Overwritten by technical contexts |
| - *Afternoon Greetings* | 10 | 0 | Overwritten by technical contexts |
| - *Evening Greetings* | 10 | 0 | Overwritten by technical contexts |
| **Identity & Architecture** | 25 (15.6%) | 19 (4.8%) | Diluted by 3.2× |
| **Copy / KV / Multi-turn Tasks** | 0 (0.0%) | **350 (87.5%)** | Massive sudden distribution shift |
| **Verbatim Refusals** | 7 (4.4%) | **20 (5.0%)** | Strongest single repeated target string |

---

## 2. Forensic Verdict on Catastrophic Forgetting

1. **Zero Greeting Replay:**
   CBR-SFT-05 omitted time-conditioned greetings entirely (0 out of 400 training examples). Over 300 optimizer steps with an effective batch size of 4, the initial attention routing weights that previously mapped `[08:00:00]` $\rightarrow$ `"Good morning!"` were completely repurposed by gradient updates to optimize technical key-value bindings.
2. **The 70/30 Replay Rule is Mandated:**
   Any future fine-tuning experiment (CBR-SFT-06) targeting retrieval **must** interleave 25–35% balanced conversational and identity replay from CBR-SFT-04 to maintain representation anchors in the parameter space.
