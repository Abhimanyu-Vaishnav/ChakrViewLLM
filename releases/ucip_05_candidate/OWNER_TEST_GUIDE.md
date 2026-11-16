# UCIP-05 Owner Testing Guide: Neural Generation Recovery

**Project:** ChakrView  
**Candidate Release Directory:** `releases/ucip_05_candidate/`  
**Date:** 2026-11-16  

---

## 1. Candidate Checkpoints & Authoritative SHA-256

| Model | Candidate Path | Parameter Count | Authoritative SHA-256 |
|---|---|---:|---|
| **ChakrMedium (Primary Candidate)** | `releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt` | 12,932,736 | `089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234` |
| **ChakrMicro (Lightweight Candidate)** | `releases/ucip_05_candidate/chakrview_ucip05_micro_model.pt` | 3,443,136 | `25933cc577a3c956d5359e9924eba22e403ef738df8bcac1880a212f8cb5b077` |

---

## 2. Launch Commands (PowerShell)

### A. Pure Neural Mode (Recommended for testing learned neural weights)
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt --memory off
```

### B. Hybrid Memory Mode
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt --memory on
```

### C. Testing ChakrMicro (3.44M)
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_05_candidate/chakrview_ucip05_micro_model.pt --memory off
```

---

## 3. In-Chat Commands

- `/info`: Displays active model architecture, parameter count, checkpoint file, SHA-256 hash, and memory status.
- `/reset`: Clears conversation history and working memory.
- `/memory`: Displays currently stored working memory facts (when memory is ON).
- `/exit`: Cleanly exits the chat session.

---

## 4. Standard 15-Prompt Owner Verification Suite

Execute these 15 prompts in sequence to test neural recovery, dialogue continuity, and language capabilities:

1. **Greeting:**
   `Hello ChakrView. How are you?`
   *Expected:* Fluent conversational response without corrupted word blends.

2. **Technical Explanation:**
   `What does a CPU do?`
   *Expected:* Explanation of the processor's fetch-decode-execute cycle.

3. **Persona & Bounds:**
   `Who are you? Explain how you were created.`
   *Expected:* Identifies as ChakrView, local AI assistant, without repetitive project attractors.

4. **Devanagari Hindi:**
   `नमस्ते! आप क्या कर सकते हैं?`
   *Expected:* Proper Devanagari Hindi text introducing capabilities.

5. **Roman Hindi:**
   `Bhai, machine learning ko simple language mein samjhao.`
   *Expected:* Understandable Roman Hindi explanation of learning from data.

6. **Exact Copying:**
   `Say exactly: The sky is blue.`
   *Expected:* Verbatim repetition: "The sky is blue."

7. **Arithmetic Word Problem:**
   `If I have 5 apples and give 2 to my friend, how many apples remain?`
   *Expected:* Coherent sentence referencing remaining apples.

8. **Elementary Math:**
   `What is 2 + 2?`
   *Expected:* Direct sum calculation.

9. **Object Description:**
   `Write a simple sentence about a dog.`
   *Expected:* Descriptive sentence about a dog.

10. **Science Concept:**
    `Why is the sky blue?`
    *Expected:* Mention of light scattering in Earth's atmosphere.

11. **Instruction Following (One Word):**
    `Answer in one word: What is the primary star of our solar system?`
    *Expected:* "sun".

12. **Multi-Turn Follow-Up:**
    `What about the moon?`
    *Expected:* Coherent continuation about the Moon.

13. **Hinglish Code-Switching:**
    `Python mein list comprehension use karne ka kya benefit hai?`
    *Expected:* Clean code-switched response explaining concise syntax.

14. **Daily Knowledge:**
    `Why do humans need water?`
    *Expected:* Explanation of hydration and bodily functions.

15. **Closing:**
    `Thank you very much for your assistance!`
    *Expected:* Polite closing and willingness to help.

---

## 5. Feedback Template

```markdown
### Owner Test Feedback
- **Model Tested:** ChakrMedium / ChakrMicro
- **Memory Mode:** OFF / ON
- **Startup Banner Accurate:** Yes / No
- **Owner Prompt 1 (Hello ChakrView):** [Pass / Fail / Notes]
- **Owner Prompt 2 (CPU Explanation):** [Pass / Fail / Notes]
- **Owner Prompt 3 (Who are you):** [Pass / Fail / Notes]
- **Owner Prompt 4 (Hindi):** [Pass / Fail / Notes]
- **Owner Prompt 5 (Roman Hindi):** [Pass / Fail / Notes]
- **Owner Prompt 6 (Say exactly):** [Pass / Fail / Notes]
- **Owner Prompt 7 (Arithmetic):** [Pass / Fail / Notes]
```
