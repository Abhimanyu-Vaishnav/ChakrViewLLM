# UCIP-05: Initial State & Repository Preflight Verification

**Date:** 2026-11-16  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Current HEAD:** `514fe35 fix(ucip-04r): implement architecture-aware checkpoint loading for ChakrMedium`  
**Git Status:** Clean (`nothing to commit, working tree clean`)  

---

## 1. Verified Cryptographic Checkpoint & Artifact Hashes

| Artifact Path | Expected / Authorized Role | SHA-256 Hash | Verification Status |
|---|---|---|---|
| `releases/ucip_04_candidate/chakrview_ucip04_model.pt` | UCIP-04 Micro Candidate (3.44M) | `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6` | **MATCH (MATCHES MANDATE)** |
| `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt` | UCIP-04 Medium Candidate (12.93M) | `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693` | **MATCH (MATCHES MANDATE)** |
| `chakrview/runtime/session.py` | Protected Runtime Invariant | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH (MATCHES MANDATE)** |
| `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt` | Historical v0.11 Release Model (3.44M) | `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754` | **VERIFIED** |
| `releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt` | UCIP-03 Micro Candidate (3.44M) | `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6` | **VERIFIED** |
| `experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt` | Medium 13M Parent Base Checkpoint | `19f248afbd5502372ccbb064d0d53fa41b362f391a2ad8f99e944fe0a036343d` | **VERIFIED** |
| `data/experiments/vocab_4096/vocab.json` | Canonical BPE Vocabulary (4096 tokens) | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** |
| `data/experiments/vocab_4096/merges.json` | Canonical BPE Merges | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** |

---

## 2. Direct Owner Failure Reproduction

The exact owner-observed failures were directly reproduced using `ConversationalChatCLI` with `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt` without resetting history between turns (the exact operational sequence during an interactive chat session):

| Turn # | Input Prompt | Observed Autoregressive Output | Owner Reported Output | Diagnosis |
|---|---|---|---|---|
| 1 | `Hello ChakrView. How are you?` | `Hello! I am an independents been sourtheral learning support ke liye available hain.` | *Hello! I am an independents been sourtheral learning support ke liye available hain.* | **100% Exact match.** Corrupted token blend from unaligned Hinglish phrases. |
| 2 | `What does a CPU do?` | `A computer ChakrView.` | *Your project is a computer provide karke predictions from mesational asks.* | Attractor collapse; prompt accumulation introduces previous turn tokens into context. |
| 3 | `Who are you? Explain how you were created.` | `Youmits interconversp and ready to assimporital networking from me! Yes, I am allyour project is a computer.` | *Youdre to assilyoks.* | Corrupted subwords and repetitive project attractor. |
| 4 | `नमस्ते! आप क्या कर सकते हैं?` | `Kaves ine.` | *Kaves software inks.* | Severe Hindi token collapse under accumulated multi-turn prompt. |
| 5 | `Bhai, machine learning ko simple language mein samjhao.` | `Your project is a computer and alyodr have in hame.` | *Kaves software is a computer and hoon.* | Collapse into corrupted English/Hinglish fragments. |
| 6 | `Say exactly: The sky is blue.` | `Youdr are your project is a weller.` | *Your project is a computer and ready to assi.* | Total inability to follow copy constraint under context. |
| 7 | `If I have 5 apples and give 2 to my friend, how many apples remain?` | `Youdr.` | *Your project is a tr Yes, .* | Immediate generation collapse into attractor prefix `Youdr`. |

In single-turn mode (clean context per prompt), the model generated:
- `What does a CPU do?` -> `"A computer processor fetches instructions from memory, decodes their meaning, and executes the required arithmetic or logical operations."` (Memorized from training record `User: Can you explain: how does a computer processor function\nAssistant:`)
- `Say exactly: The sky is blue.` -> `"The sun is a global network of our solar system."` (Memorized solar system target instead of copy instruction)
- `If I have 5 apples and give 2 to my friend, how many apples remain?` -> `"A doing is calledy made to stul planets ater."` (Hallucinatory nonsense)

---

## 3. Immediate Root Cause Identification

1. **Severe Dataset Memorization and Target Duplication:**
   In `experiments/cognition/ucip_04/data/ucip04_train.jsonl`, there are only 375 total training examples, but only 144 unique targets. The model was trained with heavy repetition of fixed sentence templates.
2. **Context Sensitivity Breakdown (Multi-Turn Collapse):**
   The training corpus contained 0 multi-turn conversation examples. Every training record had the exact format `User: <single query>\nAssistant: <target>`. In the interactive CLI, multi-turn dialogue history is prepended:
   ```
   User: ...
   Assistant: ...
   User: ...
   Assistant:
   ```
   The model had never been trained on multi-turn dialogue contexts. When more than one turn was prepended, attention degenerated into the closest training n-gram attractor (`Your project is...`, `Youdr...`, `Kaves...`).
3. **Absence of Instruction and Arithmetic Data:**
   The training dataset had zero arithmetic examples and zero constraint-following examples for word problems or copying (`Say exactly: ...`).
4. **CLI Startup Banner Inconsistency:**
   In `chakrview/chat.py`, line 241 hardcoded:
   `print(f"Model: ChakrMicro Conversational v1.0 ({self.checkpoint_path.name})")`
   regardless of whether ChakrMicro or ChakrMedium was loaded.

---

## 4. Hardware Environment & Constraints
- CPU-first local architecture.
- System RAM: 32 GB.
- Python 3.12 via `.venv\Scripts\python.exe`.
- PyTorch CPU execution validated (`torch.cuda.is_available() == False`).
