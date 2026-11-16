# CR-EXP-21: Root Cause Analysis — The Multi-Subword Attention Deficit

**Mission ID:** CR-21  
**Phase:** Phase D — Failure Diagnosis & Mechanistic Attention Tracing  
**Date:** 2026-11-15  
**Investigator:** Antigravity Autonomous Coding Agent  

---

## 1. Executive Diagnostic Finding: The Multi-Subword Attention Deficit

Prior reports attributed multi-turn fact retrieval failures either to a "defective benchmark evaluator" (CR-20R) or to "general architecture capacity" (CR-19).

Through mechanistic layer-by-layer attention tracing on `chakrview_v0101_model.pt`, we have isolated the **precise root cause**:

### Mechanistic Attention Trace (Target Prompt: `"User: My name is Aarav... User: What is my name?\nAssistant: Your name is"`):
1. **Token Fragmentation:**
   - The entity `"Aarav"` is split by Chakr-BPE into **3 subwords**: `[68 ('A'), 280 ('ar'), 2306 ('av')]`.
   - The entity `"Tanvi"` is split into: `[87 ('T'), 286 ('an'), 615 ('vi')]`.
2. **Attention Distribution of Last Token (`'is'`):**
   - The query token `'is'` attends predominantly to immediate local tokens (`pos 54 ('me ')` at $14.8\%$, `pos 52 ('ur ')` at $17.5\%$) and prompt structural tokens (`pos 44 ('\n')` at $48.1\%$).
   - Across all 6 attention layers, **attention to the entity tokens `[A, ar, av]` never exceeds $14.4\%$ in any individual head** (Layer 0: 12.8%, Layer 1: 0.3%, Layer 2: 14.2%, Layer 3: 14.4%, Layer 4: 6.0%, Layer 5: 7.8%).
3. **Severe Entity Vocabulary Scarcity in Training:**
   - In `recovery_train.jsonl`, **only 12 unique names** were used across all 1,000 dialogues.
   - The model saw the sentence frame `"Your name is..."` hundreds of times, but because only 12 names were present, the model did not learn an **in-context pointer/induction head** mechanism. Instead, it memorized an average embedding prior over the 12 names, predicting high-frequency consonant and vowel subwords (e.g. `'l'`, `'lan'`, `'h'`) rather than routing attention backwards to the specific tokens in the user prompt.

---

## 2. Why Single-Token Key-Value Retrieval Succeeded in CR-16 but Failed in Conversation

- In **CR-16 (Algorithmic Key-Value Retrieval)**, keys and values were **single atomic synthetic tokens** (`KEY_17: VAL_42`), and the training loss directly rewarded 1-step copying of that atomic token.
- In **Conversational Dialogues (CR-17 through CR-20R)**:
  1. Entities are **multi-subword natural language strings** (2–4 tokens per entity).
  2. Loss is computed on the entire sentence (`"Your name is Aarav."`), where 7 out of 11 supervised tokens are generic syntax (`['Y', 'o', 'ur ', 'na', 'me ', 'is ']`).
  3. The model achieves $90\%+$ loss reduction simply by perfecting the generic prefix, treating the actual entity tokens as a low-probability tail.

---

## 3. The Definitive Solution for CR-21

To break this deficit and produce genuine conversational memory in **v0.11**:
1. **Dynamic Entity Corpus Expansion:** Expand entity pools from 12 names/cities to **over 250 diverse names, cities, projects, tools, and technical parameters** across English, Devanagari Hindi, Roman Hindi, and Hinglish.
2. **Targeted Entity-Focused Loss Weighting (or Entity Copying Supervisions):** Ensure the model encounters diverse single-turn, 2-turn, and 4-turn dialogues where the *only* way to minimize validation loss is to attend to and copy the specific multi-subword entity from the prompt.
3. **Conversational Curriculum:** Retain the verified 25.0% replay to protect identity, greetings, and multilingual script fluency.

