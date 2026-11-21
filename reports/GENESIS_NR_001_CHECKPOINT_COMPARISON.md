# GENESIS-NR-001: Controlled Checkpoint Comparison

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  

---

## 1. Comparative Evaluation Across Model Lineage

Tested using identical prompt formatting, greedy decoding and temperature 0.7 sampling:

| Checkpoint Name | Lineage | Parameters | Pretraining Tokens | Free Generation Behavior on Greetings | Attractor Behavior |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`chakr_ultra_best.pt`** | Ultimate Brain | 5,177,612 | 0 | `'Goou I hoon! I hoon! I hoon!'` | **Severe attractor loop** |
| **`genesis_pretrain_best.pt`** | Project Genesis | 5,177,600 | 6.65M | Natural English/Hindi vocabulary continuation | None (open vocabulary) |
| **`genesis_pretrain_final.pt`** | Project Genesis | 5,177,600 | 6.65M | Multi-lingual sentence continuation | None |
| **`genesis_sft_best.pt`** | Project Genesis | 5,177,600 | 6.65M + SFT | `"Hello! I am doing well and ready to assist you."` | None (coherent assistant completion) |
| **`genesis_recovery_sft_best.pt`** | Genesis Recovery | 5,177,600 | 6.65M + Balanced SFT | Varied, diverse conversational replies | None |

---

## 2. Key Takeaway

`genesis_sft_best.pt` represents the strongest conversational checkpoint. The reason the user previously observed `Gou I hoon!` was that `releases/project_genesis_candidate/web_app.py` was mistakenly importing `ChakrViewUltimateAgent` (which loaded `chakr_ultra_best.pt`) instead of `ChakrGenesisAgent`.
