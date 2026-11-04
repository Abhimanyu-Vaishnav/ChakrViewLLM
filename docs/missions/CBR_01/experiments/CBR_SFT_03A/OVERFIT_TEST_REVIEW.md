# CBR-SFT-03A — TWO-EXAMPLE OVERFIT TEST REVIEW
## Distinguishing Memorization, Conditional Response Learning, and Generalization

---

## 1. Reproduction Setup & Execution

In an isolated temporary directory, the 2-example overfit test was reproduced using canonical Step 10k baseline weights, AdamW, $LR=10^{-3}$, and 50 steps:
- **Trained Example 1:**
  - Prompt: `User: What is the secret code?\nAssistant:`
  - Target: ` The secret code is 999.`
- **Trained Example 2:**
  - Prompt: `User: Who are you?\nAssistant:`
  - Target: ` I am ChakrView.`

---

## 2. Measured Evaluation Results

### A. Memorization Check (Trained Prompts)
- Prompt 1 Output: `'The secret code is 999.'` (**100% Bit-Exact Match**)
- Prompt 2 Output: `'I am ChakrView.'` (**100% Bit-Exact Match**)
- **Validation Loss:** Dropped from `8.2794` to `0.0004`.
- **Verdict: Memorization CONFIRMED.** The architecture has zero physical defect preventing exact conversational string emission.

### B. Conditional Response Check (Paraphrased Prompts)
- Paraphrase 1: `User: What is the secret password?\nAssistant:`
  - Actual Output: `'The secret code is 999.'` (**Correct Conditional Mapping**)
- Paraphrase 2: `User: Tell me who you are.\nAssistant:`
  - Actual Output: `'I am ChakrView.'` (**Correct Conditional Mapping**)
- **Verdict: Conditional Routing CONFIRMED on near-distribution paraphrases.** The model learned to map lexical cues (`"password"`, `"who you are"`) to the corresponding answer basin.

### C. Out-of-Distribution Generalization Check
- Unrelated Greeting: `User: Hi, good morning!\nAssistant:`
  - Actual Output: `'I am ChakrView.'` (**False Generalization**)
- **Verdict: Mode Collapse.** When presented with an unseen prompt outside the 2 examples, the model defaults to one of the 2 memorized answer basins rather than generating open-ended dialogue.

---

## 3. Epistemic Synthesis
1. **Capacity Limit Refuted:** A 3.44M parameter model can easily output clean, fluent conversational sentences when gradient energy is focused.
2. **The Mechanism of SFT in Microscopic Models:** In small models, SFT behaves primarily as a steering mechanism among discrete attractor basins rather than a general reasoning simulator.
3. If the model is trained on 540 diverse tasks with low learning rates, gradient energy is diffused and pretraining attractors win.
4. If gradient energy is focused on a compact set of high-frequency dialogue basins, conversational output is reliably triggered.
