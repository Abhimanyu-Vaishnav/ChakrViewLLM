# CR-EXP-20R: Evaluator Forensic Report (Phase C)

**Mission ID:** CR-EXP-20R  
**Phase:** Phase C — Evaluator Forensic Audit  
**Date:** 2026-11-15  
**Data Reference:** `results/recomputed_metrics.json`  

---

## 1. Executive Forensic Findings

The reported "0/700" score in CR-EXP-20 was investigated through multi-metric re-evaluation of 300 raw conversation traces across the parent baseline and both v0.10 seeds (2026 and 3030).

### Key Insights:
1. **Script Appropriateness Jumped from 0% to 95% in Devanagari:**
   - On the parent baseline, Devanagari Hindi prompts elicited **$0.0\%$ Devanagari script** (the model replied in English because it had never seen Devanagari dialogue).
   - In v0.10, **$95.0\%$ of all Devanagari prompts elicited valid Devanagari Unicode responses**.
2. **The Cause of the 0/100 Benchmark Scores:**
   - In CR-20, the benchmarks evaluated exact substring containment of arbitrary fictional entities (`"TestUser_0"`, `"TestCity_0"`, `"परीक्षार्थी_0"`).
   - The model did not output these exact strings; instead, it responded with conversational frames (`"Your name is..."`, `"Tumhara naam..."`, `"Good morning! I am ready to help"`).
   - Because the metric was binary exact containment without normalized matching or relevance scoring, it gave a completely flat score of $0/100$, masking significant progress in script selection, EOS completion ($96\%+$), and conversational persona retention.
3. **The Real Weakness Exposed (The Devanagari Attractor Collapse):**
   - While script appropriateness reached $95\%$, conversational relevance in Devanagari dropped to **$5.0\%$** because the model generated `"आपका नाम कर रहे हैं।"` for almost every question.
   - This proves that **Devanagari conversational diversity was severely lacking in the CR-20 training dataset** (over-saturated with synthetic name-recall templates).

