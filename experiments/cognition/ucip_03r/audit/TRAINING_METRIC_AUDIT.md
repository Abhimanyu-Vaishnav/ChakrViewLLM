# UCIP-03R: Phase Nine — Training & Validation Metric Audit

**Mission ID:** UCIP-03R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Date:** 2026-11-16  

---

## 1. Executive Summary

During UCIP-03, training logs reported:
- Validation Loss: **1.7433**
- Validation Token Accuracy: **65.31%**

This gave the illusion that the model had mastered 65% of the conversational targets. However, in autoregressive generation on that exact same validation set, the model achieved only **2.24% (3/134)** exact target match.

---

## 2. Forensic Analysis of Validation Token Accuracy

Lines 144–165 of `experiments/cognition/ucip_03/training/train_ucip03.py`:
```python
mask = shift_targets != -100
if mask.sum() > 0:
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    nll = loss_fn(shift_logits, shift_targets).item()
    val_loss += nll
    val_tokens += mask.sum().item()

    preds = shift_logits.argmax(dim=-1)
    val_correct += ((preds == shift_targets) & mask).sum().item()
```

### The Teacher-Forced Artifact:
1. `val_correct` is computed via **teacher-forcing** (each token prediction $t_k$ is conditioned on the **ground-truth prefix** $t_{1 \dots k-1}$).
2. In English and Devanagari Hindi, subwords like whitespace (` `), common articles (`The`, `a`), common suffixes (`ing`, `ed`, `tion`), and punctuation (`.`, `?`) constitute $>50\%$ of target tokens.
3. Once the ground-truth prefix is fed, predicting the next subword has high baseline accuracy ($\sim 65\%$).
4. In **autoregressive inference**, however, the model conditions on its **own previous output**. As soon as step 3 predicts `'p'` instead of `'The'`, the model derails into the `"planned of the bold"` attractor and never recovers.

### Metric Distinction:
- **Teacher-Forced Token Accuracy:** **65.31%** (Measures 1-step local conditional perplexity with perfect ground truth).
- **Autoregressive Sequence Match:** **2.24%** (Measures true multi-step generative capability).

Mistaking teacher-forced token accuracy for conversational fluency was a critical metric misinterpretation.
