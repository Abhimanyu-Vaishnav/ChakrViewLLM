# NR-045: Dataset Forensics Report

## 1. Executive Summary

This forensic investigation audited the training corpus (`data/corpus_v1/master_corpus_train.jsonl`, 5,826 samples) used to fine-tune **ChakrView 1.0 RC1**.

The audit uncovered **two critical dataset defects** that directly explain why RC1 generates disjoint, malformed, and repetitive outputs.

---

## 2. Critical Defect 1: Prompt Template Duplication in 82.8% of Training Samples

### The Root Cause:
In historical Stage-D and UCIP-14R datasets, prompt strings already contained explicit prompt framing, for example:
`"User: Hello! Who are you and how can you assist me?\nAssistant: "`

When `train_conversational_sft.py` loaded these samples, it formatted them by wrapping them in redundant tags:
`formatted = f"<user>\n{prompt}\n<assistant>\n{response}<eos>"`

### Concrete Evidence:
What was actually tokenized and fed into the neural network during training:
```
<user>
User: Hello! Who are you and how can you assist me?
Assistant: 
<assistant>
Hello! I am ChakrView, a lightweight sovereign neural assistant...
```

- **Impact:** **4,826 out of 5,826 training samples (82.8%)** contained this double-tagging anomaly.
- **Inference Mismatch:** At runtime, `interactive_chat.py` and `brain_runtime.py` fed prompts formatted as:
  `<user>\nHello! Who are you and how can you assist me?\n<assistant>\n`
- **Result:** The model had never seen prompts without the inner `"User: ...\nAssistant: "` sequence. Receiving the runtime prompt acted as an out-of-distribution prefix, instantly triggering out-of-distribution token hallucinations (e.g., generating internal variable names or `"Assistant: "` artifacts).

---

## 3. Critical Defect 2: High Concentration of Synthetic Boilerplate & Short Token Echoes

### Concrete Evidence:
1. **1,126 out of 5,826 samples (19.3%)** consist of trivial 1-word or 2-word copy/echo tasks (Category `COPY`):
   - Record `CBR10-000001`: `User: Neeche diye gaye token ko exact copy kijiye: EPSILON-752-Q` $\rightarrow$ `EPSILON-752-Q`
   - Record `CBR10-000002`: `User: Copy the following token exactly without any modification: ALPHA-529-E` $\rightarrow$ `ALPHA-529-E`
2. **Synthetic Boilerplate Collisions:**
   - Hundreds of responses in `master_corpus_train.jsonl` were generated using identical synthetic prefixes:
     - `"Summary #..."` (100 samples)
     - `"A remainder of..."` (100 samples)
     - `"उत्तर #..."` (100 samples)
3. **Short Response Length:**
   - The average response length across the entire training dataset is only **12.5 words**.
   - Because the dataset lacks multi-paragraph explanatory reasoning and diverse open-domain dialogues, the 17M model collapsed onto high-frequency token clusters (e.g. `"is the sum of"`, `"API development"`, `"Station 364"`).

---

## 4. Section Verdict
`DATASET DEFECT CONFIRMED: 82.8% PROMPT FORMATTING DUPLICATION AND 19.3% SHORT-TOKEN COPY BOILERPLATE DISTORTED MODEL ATTENTION AND GENERATION BOUNDARIES.`
