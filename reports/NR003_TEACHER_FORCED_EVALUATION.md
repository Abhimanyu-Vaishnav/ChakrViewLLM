# NR003: Teacher-Forced Evaluation

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  

---

## 1. Next-Token Cross-Entropy Loss on Ground Truth Text

Calculated by feeding target tokens directly into the model and measuring cross-entropy loss and perplexity:

| Held-out Evaluation Domain | Baseline Pretraining (`genesis_pretrain_best`) | Genesis SFT (`genesis_sft_best`) | Recovery SFT (`genesis_recovery_sft_best`) |
| :--- | :---: | :---: | :---: |
| **English Prose** (`"August is the eighth month of the year..."`) | Loss = 8.3240 (PPL = 4,121.8) | **Loss = 6.6536 (PPL = 775.6)** | Loss = 6.7234 (PPL = 831.6) |
| **Hindi Prose** (`"सूर्य का प्रकाश पेड़-पौधों के लिए अत्यंत आवश्यक है।"`) | Loss = 9.8097 (PPL = 18,208.8) | **Loss = 4.3067 (PPL = 74.2)** | Loss = 6.7509 (PPL = 854.8) |
| **Hinglish Dialogue** (`"User: Kaise ho bhai?\nAssistant: Main theek hoon..."`) | Loss = 9.2440 (PPL = 10,341.9) | Loss = 6.7149 (PPL = 824.6) | **Loss = 6.5968 (PPL = 732.7)** |
| **SFT Dialogue Target** (`"User: Hello! Who are you?\nAssistant: Hello! I am doing well..."`) | Loss = 8.0584 (PPL = 3,160.1) | **Loss = 3.5290 (PPL = 34.1)** | Loss = 4.2979 (PPL = 73.5) |

---

## 2. Key Diagnostic Takeaways

- Pretraining perplexity on real English/Hindi text is extraordinarily high (>4,000 PPL), confirming that the model did not learn coherent language modeling because of the token permutation error during Stage-C pretraining.
- SFT substantially improved perplexity on Hindi and dialogue targets, indicating strong capacity to learn language when token alignment is consistent.
