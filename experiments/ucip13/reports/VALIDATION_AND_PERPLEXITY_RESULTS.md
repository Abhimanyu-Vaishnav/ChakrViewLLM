# UCIP-13: Validation & Perplexity Results Report

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Validation Corpus:** `data/tokenized/stage_c/validation/` (465,954 tokens)  

---

## 1. Validation Performance Analysis

The validation partition comprises 2 binary shards with 378 clean documents across English, Hindi, Python code, Sanskrit, and GSM8k reasoning.

- **Initial Validation Loss (Step 0):** **8.3084** (Perplexity = **4,057.62**)
- **Final Validation Loss (Step 800):** **4.6859** (Perplexity = **108.41**)
- **Absolute Loss Improvement:** $\Delta \mathcal{L} = -3.6225$ nats
- **Relative Uncertainty Reduction:** **97.33%** reduction in next-token perplexity.

---

## 2. Epoch-by-Epoch Progress Graph

```text
Step   0 [########################################] Val Loss: 8.3084 (PPL: 4057.6)
Step  50 [##################################      ] Val Loss: 7.1541 (PPL: 1279.4)
Step 100 [###############################         ] Val Loss: 6.6239 (PPL:  752.9)
Step 200 [#############################           ] Val Loss: 6.3261 (PPL:  559.0)
Step 400 [########################                ] Val Loss: 5.0917 (PPL:  162.7)
Step 600 [######################                  ] Val Loss: 4.7660 (PPL:  117.5)
Step 800 [#####################                   ] Val Loss: 4.6859 (PPL:  108.4)
```

The model continuously generalized to unseen validation sequences without plateauing or diverging.
