# CHAKRVIEW OWNER CONVERSATION TEST GUIDE (UCIP-02 UPDATED)
## Hybrid Intelligence Verification & Direct Interactive Test Scenarios

**Release Candidate:** ChakrView v1.0 Experimental Hybrid Conversational Prototype  
**Checkpoint Path:** `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt`  
**Execution Environment:** 100% Local CPU Execution  

---

## 1. Launching Interactive Sessions

You can run the terminal in either **Hybrid Mode** (recommended) or **Pure Neural Mode** to personally observe the difference:

### A. Hybrid Mode (Default — Neural Model + Working Memory)
```bash
python -m chakrview chat --checkpoint releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt --memory on
```

### B. Pure Neural Mode (Memory completely disabled)
```bash
python -m chakrview chat --checkpoint releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt --memory off
```

### C. Debug Prompt Mode (Shows exact prompt tokens passed to the transformer)
```bash
python -m chakrview chat --checkpoint releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt --debug-prompt
```

---

## 2. Interactive In-Session Commands
- `/info`: Displays active model parameter count, context ceiling, prompt budget, and whether memory is ON or OFF.
- `/memory`: Displays currently stored working memory facts, entities, and update history.
- `/reset`: Flushes conversation history and clears all working memory.
- `/exit`: Exits the session.

---

## 3. Direct Test Scenarios for the Project Owner

Follow these exact steps in your terminal:

### Scenario 1 — Name Introduction & Fact Recall (Hybrid Mode)
1. Launch with `--memory on`.
2. **User >** `My name is Rahul.`
   - *Expected Response:* Acknowledges introduction.
3. **User >** `What is my name?`
   - *Expected Response:* `[Hybrid Memory: Name] Your name is Rahul.` (Exact recall without hallucination).

### Scenario 2 — Project Fact Recall & Fact Correction (Reassignment)
1. **User >** `My project is Yatrajivi, but I renamed it to ChakrView.`
   - *Verify with:* `/memory`
   - *Displays:* `Project: ChakrView (Turn X)`
2. **User >** `What is my project called now?`
   - *Expected Response:* `[Hybrid Memory: Project] Your project is ChakrView.` (Supersedes the outdated project name).

### Scenario 3 — City & Location Recall
1. **User >** `I live in Bengaluru.`
2. **User >** `Where do I live?`
   - *Expected Response:* `[Hybrid Memory: City] You live in Bengaluru.`

### Scenario 4 — Neural Persona & Self-Identification
1. **User >** `Who are you?`
   - *Expected Response:* Identifies as ChakrView neural model.

### Scenario 5 — Reset & Session Privacy
1. **User >** `/reset`
   - *Displays:* `[Conversation history and working memory reset.]`
2. **User >** `/memory`
   - *Displays:* `Working memory is currently empty.`
3. **User >** `What is my name?`
   - *Expected Response:* Informs user that no name is known in this clean session.

### Scenario 6 — Neural-Only Comparison (Contrast Test)
1. Launch with `--memory off`.
2. **User >** `My name is Rahul.`
3. **User >** `What is my name?`
   - *Observed Neural Response:* Observe that without hybrid memory, the 3.44M transformer outputs a generic name fragment (e.g. `Sarh`), illustrating why hybrid memory integration is essential for small models.
