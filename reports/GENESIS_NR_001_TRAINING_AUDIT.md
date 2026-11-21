# GENESIS-NR-001: Training & Generalization Audit

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-001  
**Project:** ChakrView  

---

## 1. Audit of Pretraining Run

- **Corpus:** Stage-C (6,651,093 tokens across 27 binary shards).
- **Validation Corpus:** 465,954 tokens across 2 held-out shards.
- **Hardware Acceleration:** NVIDIA RTX A1000 6GB Laptop GPU (`.venv_gpu`), achieving 24,000–25,000 tokens/sec.
- **Pretraining Loss Progression:**
  - Step 100: Loss = 6.5973 (PPL = 733.09)
  - Step 300: Loss = 3.8726 (PPL = 48.07)
  - Step 500: Loss = 3.3717 (PPL = 29.13), Validation Loss = 3.5381 (Val PPL = 34.40)
  - Step 800: Loss = 3.0877 (PPL = 21.93)
- **Generalization:** Model successfully learned English and Hindi subword associations, eliminating token collapse.

---

## 2. Audit of Conversational SFT Run

- **Dataset:** 1,476 unique instruction-dialogue pairs.
- **Loss Progression:**
  - Epoch 1 (Step 50): Loss = 7.1840
  - Epoch 2 (Step 200): Loss = 0.3560
  - Epoch 3 (Step 400): Loss = 0.2069
  - Epoch 5 (Step 900): Loss = 0.1195
- **Assistant Target Alignment:** Verified 100% compliant with `-100` prompt masking.
