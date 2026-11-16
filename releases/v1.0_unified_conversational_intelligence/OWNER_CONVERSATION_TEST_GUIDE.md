# CHAKRVIEW OWNER CONVERSATION TEST GUIDE
## Interactive Multi-Turn Evaluation & Conversation Readiness

**Release Candidate:** ChakrView v1.0 (Hybrid Neural Brain Architecture)  
**Checkpoint Path:** `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt`  
**Execution Environment:** 100% Local CPU Execution  
**Command to Launch:**
```bash
python -m chakrview chat --checkpoint releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt
```

---

## Interactive Session Commands
- `/info`: Displays active model parameter count, context ceiling, prompt budget, and turn statistics.
- `/memory`: Displays currently stored conversational facts, entities, and update history.
- `/reset`: Flushes conversation history and clears all working memory.
- `/exit`: Exits the conversational session.

---

## Structured Test Scenarios

Follow these scenarios step-by-step in the interactive chat terminal to evaluate real-world conversational readiness:

### Scenario 1 — Natural Persona & Greeting
1. **User >** `Hello assistant! Who are you?`  
   *Verify:* The model identifies itself as ChakrView and introduces its indigenous neural architecture.
2. **User >** `What can you help me with today?`  
   *Verify:* The model provides an appropriate conversational continuation.

### Scenario 2 — Multilingual & Code-Switching
1. **User >** `नमस्ते! आप कैसे हैं?`  
   *Verify:* The model outputs pure Devanagari Hindi text without script bleed or ASCII escape symbols.
2. **User >** `Aapka main kaam kya hai bhai?`  
   *Verify:* The model responds in colloquial Roman Hindi / Hinglish.
3. **User >** `Can we chat in English now?`  
   *Verify:* The model smoothly switches back to English.

### Scenario 3 — Hybrid Memory & Entity Storing
1. **User >** `My name is Rohan and I live in Bengaluru.`  
2. **User >** `My project is called Yatrajivi.`  
3. **User >** `/memory`  
   *Verify:* The memory command displays:
   - Name: Rohan
   - City: Bengaluru
   - Project: Yatrajivi
4. **User >** `Can you tell me about transformers?`  
   *Verify:* The model discusses the topic without hallucinating or losing history.
5. **User >** `What is my name and which city do I live in?`  
   *Verify:* The model accurately retrieves Rohan and Bengaluru.

### Scenario 4 — Fact Correction & Superseding
1. **User >** `Actually, I renamed my project to ChakrView.`  
2. **User >** `/memory`  
   *Verify:* The memory command displays `Project: ChakrView` as active and supersedes `Yatrajivi`.
3. **User >** `What is my project called now?`  
   *Verify:* The model outputs `ChakrView` rather than the old project name.

### Scenario 5 — Missing Information & Abstention
1. **User >** `What did I eat for dinner three weeks ago?`  
   *Verify:* The model recognizes uncertainty and abstains from fabricating facts.

### Scenario 6 — Session Reset & Privacy Isolation
1. **User >** `/reset`  
   *Verify:* Terminal displays `[Conversation history and working memory reset.]`.
2. **User >** `/memory`  
   *Verify:* Working memory is completely empty.
3. **User >** `What is my name?`  
   *Verify:* The model does not retain previous session facts.
