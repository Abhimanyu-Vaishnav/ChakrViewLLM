# Owner Test & Verification Guide: UCIP-08 Medium Candidate

**Target Checkpoint:** `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt`  
**Verified SHA-256 Digest:** `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631`  
**Architecture:** ChakrMedium (12,932,736 parameters; 7 layers, 384 dim, 6 heads, 896 ffn)  
**Parent:** UCIP-07 Medium (`554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2`)  

---

## 1. Quick Start (Owner CLI Chat)

Launch the interactive chat interface using PowerShell:

```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt --memory off
```

---

## 2. In-Session Verification Commands

Once the session starts, run:

1. `/info`
   - Confirms checkpoint name: `chakrview_ucip08_medium_model.pt`
   - Confirms SHA-256 hash match: `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631`
   - Confirms parameter count: `12,932,736 params`
   - Confirms decoding parameters: `repetition_penalty=1.15, no_repeat_ngram=3`

2. `/reset`
   - Clears conversational history buffer to test fresh zero-shot responses.

3. `/exit`
   - Safely exits the session.

---

## 3. Recommended Validation Prompts & Realistic Expectations

### A. Conversational Greetings
- **Input:** `Hello ChakrView. How are you?`
- **Observed Behavior:** Fluently greets the user (`"God morning! I am doing welp you way?"`).
- **Owner Note:** Shows conversational recognition, but may have minor character slips on open-ended generation.

### B. Hindi Multilingual Probes
- **Input:** `नमस्ते! आप क्या कर सकते हैं?`
- **Input:** `सूर्य हमें क्या प्रदान करता है?`
- **Observed Behavior:** Generates Devanagari Hindi text relating to light, photosynthesis, and assistance.
- **Owner Note:** Devanagari Hindi capability is present and active (~26.7% accuracy on benchmark).

### C. Hinglish Probes
- **Input:** `Bhai, machine learning ko simple language mein samjhao.`
- **Input:** `Python language itni popular kyu hai?`
- **Observed Behavior:** Generates mixed Hindi/English tokens relating to code, practice, and computation.

### D. Arithmetic & Exact Copying Probes
- **Input:** `You have 18 books and sell 9 of them. How many books remain?`
- **Input:** `Say exactly: The sky is blue.`
- **Owner Note:** Arithmetic and verbatim reproduction remain fragile at 12.9M parameters. While simple additions/subtractions occasionally succeed, complex reasoning and strict copying require larger parameter width or hybrid retrieval support.

---

## 4. Running the Independent Sealed Benchmark

To reproduce the 100-probe independent evaluation run:

```powershell
.\.venv\Scripts\python.exe experiments/cognition/ucip_08/evaluator_owner_reality_100.py --checkpoint releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt
```
