# UCIP-02: BASELINE & RECOVERY AUDIT REPORT

**Mission ID:** UCIP-02  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Date:** 2026-11-16  
**Auditor / Lead Systems Architect:** Autonomous Lead Systems Architect  

---

## 1. Executive Summary

Mission UCIP-02 investigated the specific generation and response failures reported during the owner's manual conversational evaluation of ChakrView v1.0.

### Key Forensic Findings:
1. **The "Your project was..." Attractor:**
   - 25.8% (279/1,080) of conversational targets in `cr21_train.jsonl` were fixed syntactical templates centered around project titles and cities (e.g. `"Your project was a..."`, `"You live in..."`).
   - Essential open-domain conversational probes ("Say exactly...", "Write one simple sentence about a dog", "What is 2 + 2?") had 0 occurrences in training data.
   - When prompted out-of-distribution, the 3.44M transformer fell into its strongest prior basin: the "Your project was..." attractor.
2. **The "Your name is Sarh" Failure (Hybrid Memory Pipeline Disconnect):**
   - In `chakrview/chat.py`, `relevant = self.memory.retrieve_relevant_facts(user_input)` was called, but the extracted fact was never injected into the conversational prompt or returned to the user!
   - As a result, the model was forced to guess the name using self-attention across historical turns, hitting the 3.44M parameter binding ceiling and hallucinating name fragments (`Sarh`).
3. **The 13M Candidate Model Status:**
   - The recovered 13M model (`ucip01r_13m_recovered_best.pt`, SHA-256: `19f248af...`) successfully resolved immediate EOS termination, but suffered from severe repetition collapse on open prompts (`"Your favorite favorite favorite..."`), confirming that the 13M model requires longer foundational pretraining before it can surpass the stable 3.44M architecture.

---

## 2. Checkpoint and Artifact Lineage Verification

| Checkpoint / Artifact | File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Canonical Architecture Init** | In-Memory (Seed 42) | `c5571c9cf41d7c34b172a6bcaae4f9f7431f79a957262d04f1fb9d187766dbb7` | **PASS** |
| **Canonical Stage B Pretrained**| `experiments/batch16_21_combined/...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| **Frozen Conversational Parent**| `experiments/cbr01_sft/runs/cbr_sft_04/...` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS** |
| **Protected Runtime `session.py`**| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS** |
| **Current Tested Checkpoint (v0.11)**| `releases/v1.0_unified.../chakrview_v011_model.pt` | `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754` | **PASS** |
| **Recovered 13M Checkpoint**| `experiments/cognition/ucip_01r/...` | `19f248afbd5502372ccbb064d0d53fa41b362f391a2ad8f99e944fe0a036343d` | **PASS** |

---

## 3. Engineering Remediation & Execution Modes

We implemented full capability separation in `chakrview/chat.py`:
- Added `--memory on|off` flag.
- Added `--debug-prompt` flag.
- Integrated transparent Hybrid Memory Response dispatch: When a user introduces a personal fact (`"My name is Rahul."`), the working memory engine stores it. When the user queries that fact (`"What is my name?"`), the runtime returns `[Hybrid Memory: Name] Your name is Rahul.` with clear provenance labeling.
- Verified that in pure neural mode (`--memory off`), the exact behavior of the base transformer is preserved for scientific honesty.
