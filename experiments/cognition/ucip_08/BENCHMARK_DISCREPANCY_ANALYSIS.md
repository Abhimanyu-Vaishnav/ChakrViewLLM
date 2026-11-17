# UCIP-08: Benchmark vs Owner Reality Discrepancy Analysis

## 1. Executive Summary

In UCIP-07, the candidate model `chakrview_ucip07_medium_model.pt` reported an overall accuracy of **31.37% (251 / 800)** on the sealed 800-probe benchmark, showing an apparent $+4.87\%$ absolute gain over its parent UCIP-06 Medium ($26.50\%$).

However, when the owner tested the checkpoint directly in the interactive chat environment:
```powershell
.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt --memory off
```
the model generated severe degenerative text:
- **English Greeting**: `"I am doing welp you have a and pleassist you."` (garbled portmanteaus).
- **CPU Explanation**: `"A converts program instructions, copatience of a magnet and use lives the dogance to whitation."` (incoherent).
- **Devanagari Hindi**: `"मैं आपकी समय अमु है, एक स्वस्थित होता है और पास को निसं ChakrView हूँ।"` (syntactically fragmented, nonsensical).
- **Hinglish**: `"Machine learning ek aisi technology hai jisme computer past data se ous e, aur software ka acket pasks ko behave karke new can can sakehte hain."` (repetitive loop).
- **Arithmetic**: `"You will have 5 apples remains in total."` (wrong answer).
- **Exact Copying**: `"The golden is blus of ked ree ."` (complete failure).

**Investigation Goal**: Determine why a model that scored 31.37% on the benchmark produced incoherent responses during owner testing.

---

## 2. Root Cause Classification: The 4 Discrepancy Drivers

Our forensic trace across the entire pipeline revealed **four distinct drivers** that created the illusion of high benchmark competence:

### Driver 1: Evaluator Defect — Single Keyword & Substring Matching
In `experiments/cognition/ucip_07/evaluator_800.py`:
- In categories like English Conversation, English Explanations, Roman Hindi, and Hinglish, a response was marked **PASSED** if **ANY SINGLE KEYWORD** from `expected_keywords` appeared anywhere in the output:
  ```python
  kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
  if len(kw_hits) >= 1:
      success = True
  ```
- **Arithmetic False Positive**: On `"If there are 12 birds and 5 fly away, how many stay?"` (Expected: 7), the model generated `"You now have 17 coins in total."`. The evaluator executed:
  ```python
  pattern = rf"(?:\b|_){re.escape(kw_clean)}(?:\b|_)"
  if re.search(pattern, norm_out) or kw_clean in norm_out:
  ```
  Because the substring `"7"` is present inside `"17"`, the evaluator marked this incorrect calculation as **PASSED**!
- **Scientific Explanation False Positive**: On `"What is photosynthesis?"`, the model generated:
  `"Water eving liscose ner, water do to create oxyological eving slowly"`
  Because the word `"water"` was in the keyword list, this complete word-salad was scored as **PASSED**!
- **Devanagari Script False Positive**: In Devanagari Hindi, any output having $\ge 3$ Devanagari characters and containing the common verb `"कर"` was marked **PASSED**, even when the sentence was gibberish.

### Driver 2: Prompt Variation & Brittleness
The 800-probe benchmark expanded 500 core prompt stems using 10 programmatic prefixes (e.g., `"Please compute: "`, `"Solve this: "`, `"Answer: "`, `"Quick math: "`, etc.).
- When the owner entered the base prompt:
  `"If I have 5 apples and give 2 to my friend, how many apples remain?"`
  $\implies$ Output: `"You will have 5 apples remains in total."` (FAIL).
- But when formatted with prefix:
  `"Please compute: If I have 5 apples and give 2 to my friend..."`
  $\implies$ Output: `"You will have 3 apples remains in total."` (PASS).
The model has not learned arithmetic reasoning; it has memorized specific ngram transitions conditioned on prefix tokens.

### Driver 3: Decoding Hyperparameter Discrepancy
- **Evaluator 800 Pipeline**: Used pure greedy argmax (`temperature=0.0`, `repetition_penalty=1.0`, `no_repeat_ngram=0`).
- **Interactive Chat CLI**: Applied `repetition_penalty=1.15` and `no_repeat_ngram_size=3`.
Under repetition penalty 1.15 on an undertrained model, when the model attempts to generate common connectives or repeated subwords, the penalty suppresses the top logits and forces the argmax into low-probability tail tokens, producing words like `"copatience"`, `"dogance"`, and `"sustantical"`.

### Driver 4: Metric Inflation Decomposition
Out of the 251 probes scored as "PASSED" in the UCIP-07 benchmark:
- **Genuinely coherent / acceptable outputs**: **132 / 800 (16.50%)**
- **Spurious passes (keyword-in-garbage, substring math errors, script-only)**: **119 / 800 (14.87%)**

The true zero-shot semantic capability of UCIP-07 Medium is **~16.5%**, not 31.37%. This fully explains the discrepancy observed by the owner.
