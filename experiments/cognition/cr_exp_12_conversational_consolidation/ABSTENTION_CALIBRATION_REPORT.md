# ABSTENTION CALIBRATION & EPISTEMIC DISCRIMINATION REPORT
**Investigation:** Phase 5 Diagnostic  
**Date:** 2026-11-07T22:33:45+05:30  

---

## 1. CALIBRATION GAINS IN CR-EXP-11 AND REMAINING GAPS
- **Major Advance in CR-11**: The pathological 100% false-silence failure of CR-10 was successfully broken. Answer coverage improved from $0.0\% \to 60.0\%$, and true abstention recall reached $79.0\%$.
- **Current Deficit**: On answerable questions where the model answers, only $31.0\%$ of answers are correct. The remaining $29.0\%$ are incorrect guesses. On unanswerable questions, the hallucination rate is $21.0\%$.

---

## 2. REFINED TRAINING RATIOS FOR CR-12
To hit the release gates of $\ge 75\%$ answerable accuracy and $\le 10\%$ hallucination:
- Maintain an exact **50/50 balance** between positive retrieval pairs and unanswerable entity/attribute pairs.
- Ensure that unanswerable pairs use identical vocabulary, syntax, and sentence lengths to positive pairs so the model cannot exploit superficial token length shortcuts.

