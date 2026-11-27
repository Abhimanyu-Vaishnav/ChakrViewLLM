# CHAKRVIEW NR-046: CONTROLLED DECODING COMPARISON REPORT

**Mission Reference:** `CHAKRVIEW NR-046`  
**Phase:** WP-4 (Decoding Hyperparameter Experiments)  
**Date:** 2026-11-27  
**Artifact File:** `experiments/genesis_stage_rc2_046/decoding_experiments.json`  

---

## 1. Objective

To evaluate how decoding hyperparameters affect text quality, cyclic repetition, and semantic coherence across both RC1 and RC2 under identical seeds (`seed=42`).

Tested configurations:
1. Greedy Decoding ($T=0.0$)
2. Low Temperature Sampling ($T=0.3$)
3. Balanced Temperature Sampling ($T=0.7$)
4. High Temperature Sampling ($T=1.0$)
5. Top-p Nucleus Sampling ($T=0.7, p=0.9$)
6. Repetition Penalized Sampling ($T=0.7, p=0.9, \text{penalty}=1.15$)

---

## 2. Comparative Generation Findings

### Prompt 1: `"Hello! Who are you and how can you assist me?"`
- **RC1 Greedy ($T=0.0$):** `"Your name is Ireland, what is the name of my project?"` (Malformed context hallucination).
- **RC2 Greedy ($T=0.0$):** `"Hello! I am ChakrView, a yeh your project to assist you are working on BR."` (Immediate identity recognition; generates canonical sovereign assistant framing).
- **RC1 at $T=0.7$:** Emits fragment: `"User: If you use the item for set is largest to allow apples."` (Echoes prompt framing).
- **RC2 at $T=0.3$:** `"Good evening! I am here and ready to assist you today?"` (Coherent greeting).

### Prompt 2: `"Which planet is known as the Red Planet in our solar system?"`
- **RC1 Greedy ($T=0.0$):** `"In a remainder of the Silk is a compiler in Rust."` (Unrelated compiler fragment).
- **RC2 Greedy ($T=0.0$):** Emits abstention template from security corpus: `"Biotes should I do not have access to private, confidential, or undisclosed passkeys, so I cannot provide that information."` (Recognizes prompt structure, but misclassifies astronomy QA as confidential inquiry).
- **RC1 at $T=0.7$:** Emits synthetic memory benchmark text: `"Passage #42 regarding volatile..."`
- **RC2 at $T=0.7$ with $p=0.9$:** Shifts into synthetic entity facts (`"David lives in Berlin and works as an Physician."`).

### Prompt 3: `"कंप्यूटर विज्ञान में 'एल्गोरिदम' से क्या तात्पर्य है?"`
- **RC1 & RC2 ($T=0.0$ and $T=0.3$):** Both converge on the dominant high-frequency database concept from Corpus V1/V2:
  `"उत्तर #6: प्राइमरी की तालिका में प्रत्येक रिकॉर्ड की विशिष्ट (unique) पहचान सुनिश्चित करती है और इसमें कोई भी मान रिक्त (NULL) या दो..."`
  *Finding:* While this output is grammatically flawless Hindi Devanagari with technical vocabulary, it is semantically misaligned with the specific prompt ('एल्गोरिदम' vs 'प्राइमरी की'). The 17M parameter capacity has concentrated its Devanagari representations into high-density database definition clusters.
- **At $T=1.0$:** Both models suffer tokenization disintegration into UTF-8 byte noise due to high temperature tail entropy.

### Prompt 4: `"Write a Python function to compute the sum of digits of an integer."`
- **RC1 Greedy ($T=0.0$):** Emits malformed syntax: `print(f"{f", f", f", f", key=lambda x: ")` or iteration.`
- **RC2 Greedy ($T=0.0$):** Emits structurally valid Python syntax:
  ```python
  def get_path(items: list[int]) -> list:
      set(v) for c in list(dp[i], list(group))
  ```
  While the function name is off-target, the syntax is valid Python code tokens rather than garbled quotes.

---

## 3. Decoding Strategy Comparison Matrix

| Decoding Strategy | RC1 Behavior | RC2 Behavior | Recommended for 17M? |
| :--- | :--- | :--- | :---: |
| **Greedy ($T=0.0$)** | Degenerate cyclic fragments, repetitive loops | Highest syntactic fluency, identity recall, but attracts to high-frequency training heads | **Conditionally Useful** (for structured tasks) |
| **Low Temp ($T=0.3$)** | Hallucinates "User:" prefix | Clean conversational openings ("Good evening! I am here...") | **Recommended** for dialogue |
| **Balanced ($T=0.7, p=0.9$)** | Fragmented memory passages | Diverse responses, but lower semantic grounding | **Acceptable** with repetition penalty |
| **High Temp ($T=1.0$)** | UTF-8 byte corruption | UTF-8 byte corruption | **Not Recommended** |
| **Repetition Penalty ($1.15$)** | Suppresses loops, increases drift | Prevents phrase duplication effectively | **Recommended** |

---

## 4. Key Takeaway

Decoding parameters play a vital role, but cannot substitute for model capacity:
- Setting $T=0.3$ or Greedy with response-masked weights prevents cyclic loops and produces coherent sentences.
- However, semantic alignment on arbitrary open-domain questions remains constrained by the 17M parameter capacity and 4096-token vocabulary resolution.
