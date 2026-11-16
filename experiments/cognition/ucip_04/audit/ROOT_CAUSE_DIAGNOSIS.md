# UCIP-04: Phase One — Root Cause Diagnosis: Pre-Training Forensics

**Mission ID:** UCIP-04  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Date:** 2026-11-16  
**Auditor / Lead Engineer:** Autonomous Conversational Recovery Agent  

---

## 1. Executive Summary

Before undertaking any new training in UCIP-04, an exhaustive mechanistic diagnosis was conducted to determine why previous fine-tuning runs (CR-20, CR-21, UCIP-01, UCIP-02, UCIP-03) repeatedly collapsed into repetitive attractors (`"Your project was..."` in v0.11; `"The planned of the bold"` in UCIP-03) and achieved $<3\%$ autoregressive sequence accuracy despite high teacher-forced training/validation token accuracy ($>65\%$).

The diagnosis established four compounding technical root causes:
1. **Teacher-Forcing Exposure Bias:** The models were trained exclusively with teacher-forcing where every step $k$ receives ground-truth tokens $t_{1 \dots k-1}$. In autoregressive generation, any deviation at step 1 or 2 permanently drives the model off-manifold into familiar unigram attractor basins.
2. **Associative Memory Memorization vs. Generative Circuitry:** In UCIP-03, 170 copying examples were identical proverbs (`"Fortune favors the bold"`). The model simply memorized that `"User: ... \nAssistant:"` should be followed by proverb n-grams, rather than learning an in-context copying pointer circuit.
3. **Severe Attractor Basin Overfitting:** When an unseen prompt arrives (`"User: Say exactly: The sky is blue."`), the model matches the `"User: ..."` template prefix and immediately outputs the highest unigram probability token sequence from training: `"The planned of the bold."`.
4. **Lack of Language Continuation Pretraining:** None of the conversational fine-tuning datasets ever trained the model on raw, natural sentence continuations (`"A dog is a..."`, `"A computer processes..."`). All training was conditioned on the high-level `"User: ...\nAssistant:"` format, leaving the model's base language syntax weak and fragile.

---

## 2. Token-Level Alignment Audit

We verified the tokenization and loss masking pipeline:
- `prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)`
- `target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)`
- `input_ids = prompt_ids + target_ids`
- `labels = [-100]*len(prompt_ids) + target_ids`
- In `nn.CrossEntropyLoss(shift_logits, shift_targets)`:
  - `shift_logits = logits[:, :-1]`
  - `shift_targets = labels[:, 1:]`
  - The first supervised token is correctly aligned with the final prompt token (e.g. `":"` predicting the leading whitespace or first character of the assistant response).
- **Finding:** The PyTorch loss calculation and shifted label masking are mathematically sound. The failure is not an off-by-one indexing error. It is a data composition and distribution catastrophe.

---

## 3. The Generative Deficit Across Historical Checkpoints

Testing raw prefix completion across checkpoints without the dialogue template:
- **Stage B (`10000.pt`):** Generates algorithmic synthetic tokens (`"a shows to be a show"`). Has no conversational concepts.
- **v0.11:** Degenerates immediately into `"project d"`, `"project home day"`. Severely overfit to software engineering dialogue templates.
- **13M Recovered:** Falls into unigram repetition loops (`"is is is is..."`, `"kiya kiya kiya..."`).
- **UCIP-03 Micro:** Blends proverb fragments (`"hat at at one 20"`, `"The planned of the bold"`).

---

## 4. Controlled Architectural Prescription for UCIP-04

To recover genuine neural language generation:
1. **Corpus Redesign:** Eliminate dictionary proverbs and repetitive aphorisms entirely. Construct a multi-faceted corpus comprising:
   - Category A: Pure language continuation (teaching natural English, Hindi, and Hinglish syntax from prefixes).
   - Category B: Open-ended factual explanations (describing CPUs, photosynthesis, dogs, computers, nature).
   - Category C: Direct question-answering with diverse syntactic phrasing.
   - Category D: Explicit copying across diverse everyday words, punctuation, and structures.
2. **Autoregressive Step-Wise Validation:** Measure greedy rollout generation on held-out probes at every evaluation checkpoint, never relying on teacher-forced token loss alone.
