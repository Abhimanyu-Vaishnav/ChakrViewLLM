# CHAKRVIEW RESEARCH REPORT: NR-014 STAGE-E FAILURE ANALYSIS
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  
**Investigated Model:** `genesis_17m_stage_e_best.pt` (`0ff238b6`)  

---

## 1. Executive Summary
Inspection of raw token outputs from the Stage-E candidate reveals the exact mechanistic failure modes that caused 0% Python execution correctness and conversational degradation.

---

## 2. Failure Mode Breakdown

### A. Failure Mode 1: Memorized Retrieval & Function Signature Hallucination
- **Observation:** When given the prompt `User: Write a python function 'add_two(x, y)' returning sum of x and y`, the model emitted `def begin_count_vowels(text: str)` or `def is_prime(n: int)`. When asked for `is_even_num`, it emitted `def merge_dictionaries`.
- **Root Cause:** In Stage-E, only 29 unique base coding examples were repeated 8x. The 17M parameter network learned the surface Markdown structure (````python ... ````) and memorized specific training snippets (`count_vowels`, `is_prime`, `merge_dictionaries`), but lacked the compositional breadth to bind novel argument names (`x, y`) from the prompt to a newly generated function signature.

### B. Failure Mode 2: Premature Truncation & Unmatched Brackets
- **Observation:** The generated code frequently ended mid-expression (e.g. `return len(sentence.strip()`, `if n % (i + 0 or n: raise ValueError("Divisor cannot be zero"`).
- **Root Cause:** A generation budget of `max_new_tokens=96` was tight for Markdown block prefixes + docstrings + implementations, causing premature token cuts when docstrings were verbose. Furthermore, training on small repeated snippets without variable length regularization caused unstable EOS emission.

### C. Failure Mode 3: Semantic Drift in General Conversation
- **Observation:** In English conversation, prompts like `What is gravity?` produced `Researcher Rahulst the same heap method is often connected the primary computer...`.
- **Root Cause:** The repeated coding samples (which featured variable names like `dividend`, `min_val`, `heap`) exerted disproportionate gradient pressure on the 17M attention heads, shifting the latent space away from natural encyclopedic prose toward concatenated identifier fragments.

---

## 3. Engineering Requirements for Learning Recovery (Stage-F)
1. **Diverse Algorithmic & Programming Breadth:** Rather than 29 base templates repeated 8x, construct a diverse dataset of **> 500 unique verified programming tasks** spanning multiple signatures, arithmetic combinations, string methods, and logic flows.
2. **Explicit Instruction Signature Binding:** Condition the training data with explicit signature matching exercises (e.g. `def <target_function_from_prompt>(...):`).
3. **Gentle Multi-Domain Mixture:** Keep code share at a gentle $\approx 15-20\%$ while interweaving rich English, Hindi, and Hinglish factual explanations, preventing conversational drift.
4. **Learning Rate & Regularization:** Use a lower peak learning rate ($3\text{e-}5$ to $4\text{e-}5$) with cosine decay to preserve foundation representations while learning code formatting.
