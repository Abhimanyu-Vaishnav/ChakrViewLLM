# ChakrView UCIP-04 — Owner Interactive Testing Guide

This guide provides the exact terminal commands and prompt sequences to interactively test the newly trained UCIP-04 candidate models:
`releases/ucip_04_candidate/chakrview_ucip04_model.pt` (ChakrMicro 3.44M) and `chakrview_ucip04_medium_model.pt` (ChakrMedium 12.9M).

---

## 1. Quick Start Commands

### Recommended Mode: Hybrid Conversational Intelligence (Memory ON)
```powershell
python -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_model.pt --memory on
```

### Diagnostic Mode: Pure Neural Language Generation (Memory OFF with Debug Prompt)
```powershell
python -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_model.pt --memory off --debug-prompt
```

---

## 2. Mandatory Owner Test Sequences

### Test A — Exact Copying
- **Prompt:** `Say exactly: The sky is blue.`
- **Inspection Focus:** Verifies whether the model attempts copying without falling into `"The planned of the bold."`.

### Test B — Simple Sentence Generation
- **Prompt:** `Write one simple sentence about a dog.`
- **Expected Coherence:** A meaningful statement about a dog (e.g. `"A dog is a friendly domestic animal that loves to play and guard the home."`).

### Test C — Basic Explanation
- **Prompt:** `What does a CPU do?`
- **Expected Coherence:** An explanation describing instruction execution or data processing.

### Test D — Identity & Origin
- **Prompt:** `Who are you? Explain how you were created.`
- **Expected Coherence:** ChakrView indigenous conversational identity without hallucinated claims.

### Test E — Devanagari Hindi Dialogue
- **Prompt:** `नमस्ते! आप क्या कर सकते हैं?`
- **Expected Coherence:** Natural Devanagari Hindi text addressing user assistance.

### Test F — Roman Hindi Dialogue
- **Prompt:** `Bhai, mujhe machine learning simple language mein samjhao.`
- **Expected Coherence:** A colloquial Roman Hindi or code-switched response explaining machine learning.

### Test G — Hinglish Code-Switching
- **Prompt:** `Computer ka CPU kya kaam karta hai? Simple example ke saath samjhao.`
- **Expected Coherence:** Meaningful Hindi-English technical explanation.

### Test H — Multi-Turn Follow-Up
1. `A CPU processes instructions.`
2. `Explain this in simpler words.`

### Test I — Topic Switch
1. `Write a sentence about a dog.`
2. `Explain gravity simply.`

### Test J — Memory Separation (Neural vs Hybrid)
Execute in both `--memory off` and `--memory on`:
1. `My name is Rahul.`
2. `I live in Delhi.`
3. `What is my name and where do I live?`
*(Under `--memory on`, Hybrid Memory intercepts and guarantees 100% accurate recall).*

---

## 3. Session Control Commands
- `/info`: Displays active checkpoint path, parameter count, architecture dimensions, and active turns.
- `/memory`: Displays all extracted and stored facts in active memory.
- `/reset`: Clears conversational history and resets memory.
- `/exit`: Exits the interactive chat session.
