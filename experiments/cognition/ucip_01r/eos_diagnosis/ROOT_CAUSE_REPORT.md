# UCIP-01R: Forensic Root-Cause Analysis of 13M Premature EOS Generation

**Mission ID:** UCIP-01R  
**Target:** ChakrMedium-13M Checkpoint (`experiments/cognition/ucip_01/training/ucip01_candidate_best.pt`)  
**Date:** 2026-11-16  
**Auditor / Lead Engineer:** Autonomous Lead Systems Architect  

---

## 1. Executive Summary

In UCIP-01, training of the 12.9M-parameter candidate (`ChakrMedium-13M`) reported an anomalously low validation loss of $0.5175$ and token accuracy of $87.32\%$, but in generation inference, the model produced empty outputs by immediately generating token 1 (EOS) at position 1.

Phase One of UCIP-01R conducted diagnostic probing on logits and inspects dataset collation logic, discovering a definitive, mechanistic root cause:

**ROOT CAUSE IDENTIFIED: KEY MISMATCH IN DATASET COLLATION (`target` vs `response`)**
- In `generate_ucip01_data.py`, new synthetic examples were saved with the key `"target"`, whereas base conversational dialogues loaded from `cr21_train.jsonl` used the key `"response"`.
- In `train_ucip01.py`, line 30 executed:
  ```python
  target = dialogue.get("target", "")
  target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
  ```
- For all 964 base conversational dialogues (45.0% of the entire training corpus), `dialogue.get("target", "")` returned an empty string `""`!
- Encoding `""` with `add_eos=True` produced a target sequence of exactly `[1]` (EOS)!
- Consequently, on 45% of the training batches, the model was explicitly trained with CrossEntropyLoss to predict **EOS as the immediate next token** after every conversational prompt (including greetings, English persona queries, Hindi/Hinglish questions, and identity probes).
- The low validation loss ($0.5175$) was an artifact of the model memorizing immediate EOS termination on standard conversational prompts!

---

## 2. Empirical Verification via Logit Diagnostics

Diagnostic probing across test prompts in `diagnose_13m_eos.py` confirmed this hypothesis:
- **Conversational Prompts (where `target` was empty string `""`):**
  - Prompt: `"User: Hello assistant, who are you and how can you help me?\nAssistant:"`  
    $\to$ **EOS Probability: 0.9999 (Rank 1)**
  - Prompt: `"User: नमस्ते! आप कौन हैं?\nAssistant:"`  
    $\to$ **EOS Probability: 0.9999 (Rank 1)**
  - Prompt: `"User: Mera naam Priya hai. Aapka naam kya hai?\nAssistant:"`  
    $\to$ **EOS Probability: 0.9990 (Rank 1)**
- **Synthetic Prompts (where `target` was explicitly populated):**
  - Prompt: `"[Memory: Name: Priya, City: Pune]\nUser: What is my name and where do I live?\nAssistant:"`  
    $\to$ **EOS Probability: 0.0001 (Rank 13)**
  - Prompt: `"User: Calculate 5 + 7.\nAssistant:"`  
    $\to$ **EOS Probability: 0.0001 (Rank 20)**

The model was functioning as a gradient learner; it was trained on corrupted labels where `prompt` was followed immediately by `EOS`.

---

## 3. Mandatory Remediation Strategy

1. **Reconcile Corpus Keys:** Standardize all dataset records to ensure the assistant target response is always populated under a single consistent key (`"target"`), regardless of whether it originated from base dialogues or synthetic pairs.
2. **Loss Masking Guard:** Assert that `len(target_ids) > 1` (more than just EOS) for any training record before computing loss.
3. **Controlled Retraining of ChakrMedium-13M:** Retrain the 13M architecture with the verified, non-empty dataset and monitor both validation loss and actual autoregressive token generation.
