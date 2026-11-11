# CR-EXP-17R: Release Classification Decision

## Recommendation:
**OPTION A: RETAIN AS EXPERIMENTAL PREVIEW**

---

## 1. Justification
The decision to retain **v0.9.1 as an Experimental Preview** (and explicitly reject promotion to a full conversational model or general assistant) is supported by undeniable empirical evidence gathered in this audit:

### What Justifies Retaining as Experimental Preview (Positive Accomplishments):
1. **Refusal Sink Eradication (VERIFIED):** The model completely eliminated the pathological refusal attractor loop (`"That information is not provided in the context."`) across 100 fresh benchmark probes, 30 multi-turn stress conversations, and 30 Hindi/Hinglish probes ($0.0\%$ refusal rate).
2. **Self-Identity Realization (VERIFIED):** It consistently identifies itself as ChakrView and articulates its identity and first-principles development.
3. **Greeting and Single-Turn Politeness (VERIFIED):** Demonstrates fluent, courteous single-turn conversational greetings in English and Roman Hindi.
4. **Clean Independent Loading and Deterministic Runtime (VERIFIED):** The checkpoint loads under standard PyTorch CPU execution (`weights_only=True` with `ModelConfig`), conforms bit-for-bit to the 3,443,136 parameter ChakrMicro architecture, and executes reliably via `python -m chakrview chat`.

### What Prevents Promotion to a Higher Classification (Critical Limitations):
1. **Multi-Turn Context Saturation / Induction Failure (FAILED):**
   - In 30 multi-turn stress conversations (5–8 turns), the model achieved **0.0% fact retention** across multi-turn context boundaries.
   - It cannot reliably track names, locations, numbers, or corrections introduced 2–3 turns earlier.
2. **Hindi / Hinglish Generalization Bound (PARTIALLY VERIFIED):**
   - While Roman Hindi greetings work (`"Namaste! Main aapki kya madad kar sakta hoon?"`), Devanagari Hindi collapses into repetitive Wikipedia-style sentences (`"कथन\nयह है कि यह दोनों में मुख्य रूप से जाना जाता है।"`) rather than engaging in dialogue.
3. **Representational Bottleneck ($3.44\text{M}$ Parameters):**
   - With $L=6, d=192$, the parameter budget is fundamentally insufficient to store broad world facts or execute complex reasoning.

## 2. Conclusion
ChakrView v0.9.1 is a valid, reproducible, and meaningful milestone that successfully cures conversational degeneration and refusal loops on CPU. Retaining it strictly as an **Experimental Preview** honestly communicates its boundaries to researchers and developers while protecting scientific credibility.
