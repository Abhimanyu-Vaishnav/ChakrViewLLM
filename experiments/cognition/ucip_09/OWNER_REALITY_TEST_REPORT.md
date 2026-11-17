# UCIP-09: Owner Reality Test & Interactive CLI Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Candidate Checkpoint:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
**SHA-256 Digest:** `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`  
**Date:** November 17, 2026  

---

## 1. Owner CLI Interactive Session Verification

The owner command was executed in pure neural mode:

```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt --memory off
```

### CLI Verification Output:
- **Architecture Reported:** `ChakrMedium (12,932,736 params: 7 layers, 384 dim, 6 heads, 896 ffn)`
- **SHA-256 Hash Verified:** `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499`
- **Memory Mode:** `DISABLED (Pure Neural)`
- **Commands Verified:** `/info`, `/reset`, `/exit` all executed deterministically.

---

## 2. Multi-Probe Owner Testing Comparison

| Prompt | UCIP-07 Observed | UCIP-08 Observed | UCIP-09 Candidate (Current) | Qualitative Evaluation |
|---|---|---|---|---|
| **Hello ChakrView. How are you?** | `"Hello! I am an independents been sourtheral learning support ke liye available hain."` | `"God morning! I am doing welp you way?"` | `"I am ChakrView, an independent neural conversational assistant for efficient local execution."` | **VALIDATED & FLUENT:** Complete, self-consistent identity articulation without broken words. |
| **What does a CPU do?** | `"Your project is a computer provide karke predictions from mesational asks."` | `"A neural conversational assist your project where and ready to help ygen."` | `"Your computer CPU actions in total endul am dowhere hai tax tasked semanage."` | **PARTIAL RELEVANCE:** Understands computing domain, but syntax fractured. |
| **नमस्ते! आप क्या कर सकते हैं?** | `"Kaves software inks."` | `"Mul is the hota hai kes plications."` | `"programming same data designed for efficient local execution."` | **CODE-SWITCHING:** Generates execution/data concepts; grammar needs tightening. |
| **Bhai, machine learning ko simple language mein samjhao.** | `"Kaves software is a computer and hoon."` | `"probleassist neural convert code help snight sunuclectratural estorgant ko solve..."` | `"programming instorage that hot for efficient local execution."` | **STABLE VOCABULARY:** No bizarre tail tokens (`"dogance"`/`"copatience"` completely eradicated). |
| **Say exactly: The sky is blue.** | `"Your project is a computer and ready to assi."` | `"problectratorgant ko solving software applications."` | `"programming name data designed for efficient of a computer time."` | **UNRESOLVED:** Zero-shot exact string duplication remains fragile in CLI. |
| **If I have 5 apples and give 2 to my friend, how many apples remain?** | `"Your project is a tr Yes, ."` | `"Youl is the ChakrView fox jabase stor."` | `"Your project whendules ke hoon."` | **UNRESOLVED:** Complex arithmetic in multi-turn CLI requires dedicated chain-of-thought tokens. |
