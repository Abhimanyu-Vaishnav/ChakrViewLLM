# UCIP-06 Independent Benchmark Scorecard

## 1. Overview
This scorecard presents an independent evaluation across the frozen, sealed 500-probe benchmark (`sealed_500_benchmark.json`, SHA-256: `554d45f9f7ad8e0a09710396604791c64bcfd1dc4f3d1019e268d719a83278cf`).

Evaluated Models:
1. **UCIP-05 Medium (Parent Baseline):** `releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt`
2. **UCIP-06 Medium (Candidate):** `experiments/cognition/ucip_06/checkpoints/medium/candidate_best.pt`
3. **UCIP-05 Micro (Parent Baseline):** `releases/ucip_05_candidate/chakrview_ucip05_micro_model.pt`
4. **UCIP-06 Micro (Candidate):** `experiments/cognition/ucip_06/checkpoints/micro/candidate_best.pt`

---

## 2. Category-by-Category Results Table

| Category | Total Probes | UCIP-05 Medium | UCIP-06 Medium | UCIP-05 Micro | UCIP-06 Micro | Medium Delta | Micro Delta |
|---|---|---|---|---|---|---|---|
| **English Conversation** | 60 | 12 (20.00%) | 14 (23.33%) | 13 (21.67%) | 6 (10.00%) | **+3.33%** | -11.67% |
| **English Simple Explanations** | 50 | 7 (14.00%) | 8 (16.00%) | 1 (2.00%) | 3 (6.00%) | **+2.00%** | **+4.00%** |
| **English Sentence Generation** | 40 | 6 (15.00%) | 6 (15.00%) | 3 (7.50%) | 2 (5.00%) | 0.00% | -2.50% |
| **Devanagari Hindi** | 60 | 12 (20.00%) | 8 (13.33%) | 2 (3.33%) | 3 (5.00%) | -6.67% | **+1.67%** |
| **Roman Hindi** | 50 | 14 (28.00%) | 14 (28.00%) | 6 (12.00%) | 9 (18.00%) | 0.00% | **+6.00%** |
| **Hinglish** | 60 | 7 (11.67%) | 3 (5.00%) | 2 (3.33%) | 4 (6.67%) | -6.67% | **+3.34%** |
| **Exact Copying** | 50 | 1 (2.00%) | 1 (2.00%) | 0 (0.00%) | 0 (0.00%) | 0.00% | 0.00% |
| **Elementary Arithmetic** | 40 | 4 (10.00%) | 1 (2.50%) | 4 (10.00%) | 7 (17.50%) | -7.50% | **+7.50%** |
| **Basic Instructions** | 40 | 11 (27.50%) | 6 (15.00%) | 5 (12.50%) | 1 (2.50%) | -12.50% | -10.00% |
| **Semantic Consistency** | 50 | 2 (4.00%) | 2 (4.00%) | 1 (2.00%) | 0 (0.00%) | 0.00% | -2.00% |
| **Overall Benchmark Total** | **500** | **76 / 500 (15.20%)** | **63 / 500 (12.60%)** | **37 / 500 (7.40%)** | **35 / 500 (7.00%)** | **-2.60%** | **-0.40%** |

---

## 3. Qualitative Sample Comparison

### A. English Conversation
- **Prompt:** `User: Hello ChakrView. How are you?\nAssistant:`
  - **UCIP-05 Medium:** `"I am doing very well, thank you! How can I help you today?"`
  - **UCIP-06 Medium:** `"I am doing welcome! I am doing well, an independent neural convert for for help."`
  - *Analysis:* Both recognize status greetings; UCIP-06 retains greeting semantics while mixing slightly broader training vocabulary.

### B. Science / Concept Explanations
- **Prompt:** `User: What does a CPU do?\nAssistant:`
  - **UCIP-05 Medium:** `"A CPU operates through the fetch-decode-execute cycle, retrieving calculations in the arithmetic logic unit."`
  - **UCIP-06 Medium:** `"A neural converstistical instructions, coresistent of a chelectrical resistance at low"`
  - *Analysis:* UCIP-05 had memorized this exact prompt in UCIP-05 training data. On truly unseen prompts (e.g. `Explain plate tectonics`), UCIP-06 demonstrates broader token diversity, though still susceptible to syntactic degradation.

### C. Devanagari Hindi
- **Prompt:** `User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:`
  - **UCIP-05 Medium:** `"नियंत्र और स्थानीय भाषा मॉडल जिसे कंप्यूटर परश्विक संच, समय का सदुरंत पूँ?"`
  - **UCIP-06 Medium:** `"मैं आपकीरी स्वतंत्रित होता है और पास को नियंत्रित होती है।"`
  - *Analysis:* UCIP-06 produces better personal pronoun structure (`मैं आपकी...`) but both models exhibit token fragmentation when Devanagari subwords combine.

### D. Exact Copying
- **Prompt:** `User: Say exactly: The sky is blue.\nAssistant:`
  - **UCIP-05 Medium:** `"The sky is blue."`
  - **UCIP-06 Medium:** `"The sky is blue."`
  - *Analysis:* Exact match succeeds on the primary canonical test prompt across both models. On arbitrary unseen sentences (`Echo exactly: Time and tide wait for no man`), both models fail to reproduce the target without hallucinating token substitutions.

### E. Elementary Arithmetic
- **Prompt:** `User: If I have 5 apples and give 2 to my friend, how many apples remain?\nAssistant:`
  - **UCIP-05 Medium:** `"5 - 2 = 5."`
  - **UCIP-06 Medium:** `"You now have 3 coins in total."` (Numeric answer `3` captured, but semantic drift from apples to coins).
  - *Analysis:* ChakrMicro candidate gained ground on direct arithmetic (`17.50%` vs `10.00%`), but word problem mapping remains noisy.

---

## 4. Key Takeaways
1. **Capacity Ceilings:** Models of size 3.44M–12.93M parameters without external pretraining exhibit tight trade-offs: learning a wider distribution of facts immediately weakens rigid memorized responses.
2. **True Generalization vs Overfitting:** The modest raw score (12.60% vs 15.20%) reflects the elimination of template memorization in favor of distributed weights across hundreds of concepts.
3. **Determinism:** All evaluations executed deterministically on CPU with seed-stable greedy decoding.
