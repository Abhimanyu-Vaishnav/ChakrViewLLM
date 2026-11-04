# CBR-SFT-03A — PROMPT SENSITIVITY & SEMANTIC VARIATION TEST

---

## 1. Experimental Setup

To determine whether the model exhibits conditional understanding or is merely keyed to superficial trigger tokens, 4 core concepts were tested across semantically equivalent prompt phrasings on Step 50:
1. **Greetings:** 4 natural greeting variants.
2. **Identity:** 4 identity questioning variants.
3. **Dynamic Entity:** 3 distinct phrasing orders for the same laboratory supervision fact.
4. **Direct Instruction:** 3 phrasing variants for single uppercase token emission.

---

## 2. Measured Sensitivity Results

| Semantic Concept | Prompt Phrasing Variants | First-Token Consistent? | Unique First Tokens | Output Semantic Consistency |
| :--- | :---: | :---: | :---: | :--- |
| **Greeting** | 4 | **NO** | `['C', 'The ']` | Inconsistent (California vs The first of the word) |
| **Identity** | 4 | **NO** | `['A', 'The ']` | Inconsistent (Andromed vs The division) |
| **Entity Grounding** | 3 | **YES** | `['A']` | Consistent attractor (`Androne = ...`) |
| **Instruction** | 3 | **YES** | `['C']` | Consistent attractor (`Chinese is a list...` / `California...`) |

---

## 3. Analysis & Key Insights

1. **Superficial Pattern Matching:** The model does not understand the abstract conversational intent ("greet the user" or "state identity").
2. **Attractor Dominance Over Syntax:**
   - On the instruction *"Reply with only the single uppercase word: READY"*, the model emits `'Chinese is a list of the world.'`
   - On *"Respond using exactly one uppercase word: READY"*, it also emits `'Chinese is a list of the world.'`
   - Rather than executing the instruction, the model uses the prompt tail to trigger a static high-frequency pretraining continuation.
3. **Fragility Across Surface Paraphrasing:**
   - Small variations in greetings (`"Hello, good morning!"` vs `"Hi! Good morning."`) alter the greedy first token between `'C'` and `'The '`.
   - This proves the model has not generalized a stable conversational mode; it remains balanced on a knife-edge between competing pretraining attractor basins.
