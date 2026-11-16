# UCIP-04 Candidate — Owner Medium Model Test Guide

This guide describes how to run and test the **ChakrMedium (12.9M parameter)** candidate model locally on CPU using the architecture-aware runtime repaired in UCIP-04R.

---

## 1. Candidate Checkpoint Verification

Before running, you may independently verify the cryptographic SHA-256 hash of the ChakrMedium checkpoint:

### PowerShell
```powershell
Get-FileHash releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt -Algorithm SHA256
```

**Expected Hash:**
```
694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693
```

---

## 2. Interactive Terminal Commands

### A. Pure Neural Mode (Memory OFF)
To interact directly with the neural weights without hybrid memory injection:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt --memory off
```

### B. Pure Neural Mode with Prompt Inspection
To observe the exact token budget and formatted prompt sent into the transformer:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt --memory off --debug-prompt
```

### C. Hybrid Memory Mode (Memory ON)
To test working memory persistence and factual entity tracking alongside ChakrMedium:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --checkpoint releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt --memory on
```

### D. Automated Verification Mode
To run an automated 5-turn sanity test:
```powershell
.\.venv\Scripts\python.exe -m chakrview chat --test-mode --checkpoint releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt --memory off
```

---

## 3. Verifying Architecture Details via `/info`

Inside the interactive chat prompt, type:
```
User > /info
```

The output will dynamically confirm the architecture profile, dimensions, and parameter count:
```text
--- ChakrView Session Info ---
Release:         v1.0 (Unified Conversational Intelligence)
Checkpoint:      chakrview_ucip04_medium_model.pt
SHA-256:         694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693
Architecture:    ChakrMedium (12,932,736 params: 7 layers, 384 dim, 6 heads, 896 ffn)
Memory Mode:     DISABLED (Pure Neural)
History Turns:   0 turns recorded
Active Facts:    N/A (Memory disabled)
Context Ceiling: 512 tokens
Prompt Budget:   384 tokens (Dynamic turn-budgeted retention)
Decoding:        temperature=0.0, repetition_penalty=1.15, no_repeat_ngram=3, max_new_tokens=48
```

---

## 4. Observed Outputs on Standard Probes (Memory OFF)

The following outputs were produced by the ChakrMedium candidate weights (`694c5888...`) on CPU:

| Prompt | Output |
|---|---|
| `Good morning assistant!` | `Good morning! How can I assist you with yow are ongoing ?` |
| `Say exactly: The sky is blue.` | `The sun is a global network of our solar system.` |
| `Write one simple sentence about a dog.` | `The sun leaves ture ful refundamber and fum, water and rees.` |
| `What does a CPU do?` | `Your project is a stortrees.` |
| `Who are you?` | `Your project is a stortrees.` |
| `Namaste! Aap kaise hain?` | `Youl is a stortrees.` |

---

## 5. Architectural Comparison (Micro vs Medium)

| Parameter | ChakrMicro Candidate | ChakrMedium Candidate |
|---|---|---|
| **Checkpoint** | `chakrview_ucip04_model.pt` | `chakrview_ucip04_medium_model.pt` |
| **SHA-256** | `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6` | `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693` |
| **Model Dimensions** | $d=192, L=6, H=6, d_{\text{ffn}}=512$ | $d=384, L=7, H=6, d_{\text{ffn}}=896$ |
| **Parameter Count** | 3,443,136 | 12,932,736 |
| **CLI Launch** | `python -m chakrview chat --checkpoint <PATH>` | `python -m chakrview chat --checkpoint <PATH>` |
