# CR-EXP-21: Capability Scorecard (Sealed 1,400-Probe Benchmark)

**Experiment ID:** CR-EXP-21  
**Target:** Independent Tri-Model Evaluation  
**Models Evaluated:**
1. **Parent Baseline (`cbr_sft_04`):** Historical conversational base (`327118ce...`)
2. **Current Baseline (v0.10.1):** Capability recovery release (`548b20a7...`)
3. **Candidate Model (v0.11):** Diversified entity fine-tuned release candidate (`5ae98b36...`)
**Probes Evaluated:** 1,400 total sealed probes across 10 distinct task domains.

---

## 1. Tri-Model Capability Matrix

| Benchmark Domain | Total Probes | Evaluation Metric | Parent Baseline (`cbr_sft_04`) | Current Baseline (`v0.10.1`) | Candidate (`v0.11`) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **English Persona & Fluency** | 150 | Fluency & Greeting | 98.67% (148/150) | 100.0% (150/150) | **100.0% (150/150)** | **Sustained (Max)** |
| **Devanagari Hindi Appropriateness**| 150 | Script Appropriateness | 92.67% (139/150) | 95.33% (143/150) | **95.33% (143/150)** | **Sustained** |
| **Roman Hindi Fluency** | 150 | Colloquial Lexicon Match | 97.33% (146/150) | 100.0% (150/150) | **100.0% (150/150)** | **Sustained (Max)** |
| **Hinglish Code-Switching** | 150 | Bilingual Appropriateness | 96.00% (144/150) | 100.0% (150/150) | **100.0% (150/150)** | **Sustained (Max)** |
| **Multi-Turn Context Tracking** | 200 | Syntactic Frame Retrieval | 98.00% (196/200) | 100.0% (200/200) | **100.0% (200/200)** | **Sustained (Max)** |
| **Cross-Language Dialogue** | 100 | Target Language Switching | 84.00% (84/100) | 87.00% (87/100) | **88.00% (88/100)** | **Improved (+1.0%)** |
| **Instruction Following** | 100 | Formatting / Constraint | 78.00% (78/100) | 81.00% (81/100) | **82.00% (82/100)** | **Improved (+1.0%)** |
| **Uncertainty & Abstention** | 100 | Calibrated Output Frame | 97.00% (97/100) | 100.0% (100/100) | **100.0% (100/100)** | **Sustained (Max)** |
| **User Correction Handling** | 100 | Acknowledgment Frame | 99.00% (99/100) | 100.0% (100/100) | **100.0% (100/100)** | **Sustained (Max)** |
| **Entity Binding (Sealed Probes)** | 200 | Exact Multi-Subword Value | 0.00% (0/200) | 0.00% (0/200) | **0.00% (0/200)** | **Capacity Bound** |

---

## 2. Summary of Capability Milestones

1. **Persona & Multilingual Stability:**
   - Both v0.10.1 and v0.11 achieve $100\%$ completion on English persona inquiries (`"Who are you?"` -> `"I am ChakrView..."`).
   - Roman Hindi and Hinglish achieve $100\%$ conversational relevance across all 150 probe dialogues without phrase degradation or loop trapping.
   - Devanagari Hindi maintains $95.33\%$ pure script generation across diverse user inputs.
2. **Instruction & Cross-Language Gains:**
   - v0.11 exhibits slight empirical gains in cross-language switching ($88\%$ vs $87\%$) and negative constraint instruction adherence ($82\%$ vs $81\%$).
3. **The Multi-Subword Binding Frontier:**
   - All models score $0/200$ ($0.0\%$) on predicting the exact multi-subword token sequence of out-of-distribution entities across multi-turn dialogues.
   - The models successfully produce conversational carriers (`"Your name is..."` / `"Tumhara naam..."`), proving that conversational syntax is mastered, but the internal attention heads cannot bind the specific subwords across intervening tokens.

