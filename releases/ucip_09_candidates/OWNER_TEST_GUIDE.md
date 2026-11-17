# Owner Test & Verification Guide: UCIP-09 Medium Candidate

**Target Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
**Verified SHA-256 Digest:** `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`  
**Architecture:** ChakrMedium (12,932,736 parameters; 7 layers, 384 dim, 6 heads, 896 ffn)  
**Parent Checkpoint:** UCIP-08 Medium (`fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631`)  

---

## 1. Launching the Interactive Chat Interface

Open PowerShell in `D:\Project\ChakrView` and execute:

```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt --memory off
```

---

## 2. In-Session Verification Commands

1. `/info`
   - Verifies checkpoint name: `chakrview_ucip09_medium_model.pt`
   - Verifies SHA-256 hash: `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`
   - Confirms parameter count: `12,932,736 params`
   - Confirms architecture dimensions: `7 layers, 384 dim, 6 heads, 896 ffn`
2. `/reset`
   - Clears conversational history buffer before testing new zero-shot inputs.
3. `/exit`
   - Cleanly closes the terminal session.

---

## 3. Recommended Validation Prompts & Realistic Behaviors

### Test 1: Identity & Conversational Opening
- **Input:** `Hello ChakrView. How are you?`
- **Observed Candidate Output:** `"I am ChakrView, an independent neural conversational assistant for efficient local execution."`
- **Owner Observation:** Generates complete, grammatically sound identity without broken words.

### Test 2: Hindi Multilingual Dialogue
- **Input:** `नमस्ते! आप कौन हैं?`
- **Observed Candidate Output:** `"नमस्ते! मैं आपका स्वागत है।" / "नमस्ते! मैं चार्क है, आपका सहायका..."`
- **Owner Observation:** Devanagari Hindi generation active (achieves 38.6% on 500-probe suite).

### Test 3: Technical Hinglish Query
- **Input:** `Bhai, machine learning ko simple language mein samjhao.`
- **Owner Observation:** Tail-token hallucinations (`"dogance"`, `"copatience"`) are completely eradicated. Generates technical computing tokens.

### Test 4: Word-Problem Arithmetic
- **Input:** `If you have 15 books and give away 7 of them, how many books remain?`
- **Owner Observation:** Demonstrates entity tracking, avoiding the monolithic `"coins in total"` template observed in earlier versions.
