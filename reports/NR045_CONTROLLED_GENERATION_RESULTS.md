# NR-045: Controlled Generation & Teacher-Forcing Results

## 1. Executive Summary

This report documents Phase 6 and Phase 7 empirical testing on the frozen **ChakrView 1.0 RC1 Checkpoint** (`CF585EBF...`) under varied decoding temperatures and teacher-forced validation evaluation (`experiments/nr045_generation_forensics/generation_and_teacher_forcing_results.json`).

---

## 2. Controlled Temperature Experiments ($T \in \{0.0, 0.3, 0.7, 1.0\}$)

| Prompt Category | Temperature ($T=0.0$, Greedy) | Temperature ($T=0.7$) | Temperature ($T=1.0$) |
| :--- | :--- | :--- | :--- |
| **English Factual** (*"Which planet is known as the Red Planet?"*) | `'The sum of planet is the sum of bag and 51 is the sum of blu'` | `'Mars is called the Red Planet because from an in, other on Mars'` | `'Mars is the that and for Red Planet is because from an in'` |
| **English Code** (*"Write a Python function to add two numbers."*) | `'```python\ndef get_path(items: list[int]) -> bool:\n setattr'` | `'```python\ndef add(a: int, b: int) -> int:\n return a + b'` | `'```python\ndef add(a: int, b: int) -> int:\n return a + b'` |
| **Hindi Science** (*"कंप्यूटर विज्ञान में 'एल्गोरिदम' से क्या तात्पर्य है?"*) | `'उत्तर #6: प्राइमरी की तालिका में प्रत्येक रिकॉर्ड की विशिष्ट'` | `'एल्गोरिदम एक सुपरवाइज्ड और अनसुपरवाइज्ड लर्निंग का अर्थ है'` | `'प्रस्तुत संदर्भ के अनुसार, एल्गोरिदम किसी विशिष्ट समस्या को'` |
| **Hinglish Dialogue** (*"CPU aur GPU me main difference kya hota hai?"*) | `'Unit test programming aur system takeaway yeh hai: how is a '` | `'CPU general-purpose tasks ke liye hota hai jabki GPU parallel tasks'`| `'CPU serial processing karta hai aur GPU parallel computing ke liye'`|
| **Arithmetic** (*"What is 12 + 15?"*) | `'12<eos>\n12<eos>\n24 + 12 = 120.<eos>\nThe sum of 124 is the to'` | `'The sum of 12 + 15 is 27.<eos>'` | `'The answer is 27.'` |
| **Logic** (*"All humans are mortal. Socrates is a human. Is Socrates mortal?"*) | `'A books\nAssistant: (Bili 27): Practice is a wonderful city. '` | `'Yes, Socrates is mortal because all humans are mortal.'` | `'Yes, since Socrates is human, he is mortal.'` |

### Key Decoding Finding:
- **Greedy Decoding Failure:** Under strict greedy decoding ($T=0.0$), the model collapses onto dominant repetitive training phrases (`"The sum of..."`, `"उत्तर #6"`).
- **Sampling Recovery:** At $T=0.7$, the model generates substantially more accurate and coherent answers on factual questions, Python code, and logic.
- **Root Cause of Apparent Degeneracy in Previous Runs:** Previous runtime scripts ran at $T=0.0$ (argmax) without repetition penalties, forcing the model into degenerate low-entropy loops.

---

## 3. Teacher-Forced vs. Free-Running Evaluation

Evaluated across 25 representative validation records from `data/corpus_v1/master_corpus_val.jsonl`:

- **Teacher-Forced Validation Loss:** **0.5623** (Perplexity: **1.75**)
- **First-Token Match Rate:** **24 / 25 (96.0%)**  
  *Finding:* The model predicts the exact reference opening token with 96% accuracy under teacher forcing.
- **Average Divergence Step:** **7.52 tokens**  
  *Finding:* In free-running autoregressive generation, without the ground-truth prefix being fed back at each step, the model diverges from the target reference within 7.5 tokens on average. Because the 17M model has limited capacity, once an autoregressive error occurs, subsequent tokens rapidly drift.
