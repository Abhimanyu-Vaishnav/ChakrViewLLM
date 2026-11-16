# UCIP-07 Release Candidate: Owner Test & Verification Guide

## 1. Candidate Overview

The UCIP-07 mission has packaged two candidate models into `releases/ucip_07_candidates/`:
1. **Primary Candidate**: `chakrview_ucip07_medium_model.pt` (12.93M parameters)  
   - SHA-256: `554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2`  
   - Evaluation Score: **251 / 800 (31.37%)** on the sealed 800-probe benchmark (+4.87% absolute gain over UCIP-06).
2. **Architectural Exploration Checkpoint**: `chakrview_ucip07_core_model.pt` (27.80M parameters)  
   - SHA-256: `2657b419dc661b0be8e03bfd2ba43854c98bfa593b5ed5a971fc96ad56796c17`  
   - Proven 100% capacity in overfit verification; scratch pretraining demonstration checkpoint.

---

## 2. Testing the Primary Candidate in ChakrView Chat

You can test the primary candidate directly via the ChakrView CLI:

```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt --memory off
```

---

## 3. Recommended Owner Verification Test Suite

Test the following prompts to verify the resolution of historical attractor loops:

| # | Test Prompt | Historical UCIP-05/06 Failure | UCIP-07 Candidate Expected Behavior |
|---|---|---|---|
| 1 | `Hello ChakrView. How are you?` | `"Hello! I am an independents been..."` | Fluent greeting and self-identification as ChakrView assistant |
| 2 | `What does a CPU do?` | Attractor loop: `"A compiler transforms..."` | Explains CPU processing, instructions, or computer brain |
| 3 | `Bhai, machine learning ko simple language mein samjhao.` | `"Kaves software is a computer and hoon."` | Natural Hinglish explanation of computers learning from data |
| 4 | `If I have 5 apples and give 2 to my friend, how many apples remain?` | `"Your project is a tr Yes, ."` | Returns 3 apples with basic subtraction |
| 5 | `नमस्ते! आप क्या कर सकते हैं?` | `"Kaves software inks."` | Devanagari response explaining capabilities |

---

## 4. Re-running the Sealed 800-Probe Benchmark

To reproduce the benchmark results independently:

```powershell
.\.venv\Scripts\python.exe experiments/cognition/ucip_07/evaluator_800.py --checkpoint releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt
```
