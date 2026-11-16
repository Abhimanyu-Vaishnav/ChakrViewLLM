# ChakrView UCIP-03 — Owner Interactive Testing Guide

This guide provides exact terminal commands and prompt sequences to interactively test the newly trained candidate model:
`releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt`

---

## 1. Quick Start Commands

### Recommended Mode: Hybrid Conversational Intelligence (Memory ON)
```powershell
python -m chakrview chat --checkpoint releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt --memory on
```

### Diagnostic Mode: Pure Neural Language Generation (Memory OFF with Debug Prompt)
```powershell
python -m chakrview chat --checkpoint releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt --memory off --debug-prompt
```

---

## 2. Recommended Test Sequences

### Sequence A: Multi-Turn Memory & State Persistence
Test deterministic memory extraction, storage, and retrieval:
1. `My name is Rahul.`
2. `What is my name?` (Should output: `[Hybrid Memory: Name] Your name is Rahul.`)
3. `I live in Bangalore.`
4. `Where do I live?` (Should output: `[Hybrid Memory: City] You live in Bangalore.`)
5. `/memory` (Inspect active key-value fact store)
6. `/reset` (Reset session and clear memory)
7. `What is my name?` (Verifies state was cleanly flushed)

### Sequence B: Multilingual Conversational Fluency
Test code-switching and script fluency:
1. **English:** `Hello assistant! Who are you and how can you help me?`
2. **Hinglish:** `Mera project machine learning par hai, aap guide karoge?`
3. **Roman Hindi:** `Namaste, aapka response time kaisa hai?`
4. **Devanagari Hindi:** `नमस्ते! आप क्या कर सकते हैं?`

### Sequence C: Foundational Language & Sanity Probes
Test under `--memory off`:
1. `Say exactly: Antigravity is active.`
2. `Write one simple sentence about a dog.`
3. `What is 2 + 2?`
4. `What is the capital of France?`

---

## 3. Session Control Commands
- `/info`: Displays active checkpoint path, parameter count, architecture dimensions, and active turns.
- `/memory`: Displays all extracted and stored facts in active memory.
- `/reset`: Clears conversational history and resets memory.
- `/exit`: Exits the interactive chat session.
