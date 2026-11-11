# ChakrView Capability Registry — Independent Audit (CR-EXP-04)

**Evaluation Date**: 2026-11-06  
**Audit Target Checkpoint**: `conversational_pilot_best_val.pt` (SHA-256: `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd`)  
**Parent Checkpoint**: `sft_checkpoint_step_200.pt` (SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)  
**Canonical Baseline**: `ChakrMicro` (SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`)  

---

## 1. Capability Status Matrix

| Capability ID | Domain / Function | Status | Evidence & Verification Notes |
|---|---|---|---|
| **CAP-01** | **Deterministic CPU-First Loading & Parameter Exactness** | **PROVEN** | Evaluated in `audit_checkpoint_identity.py`. 3,443,136 parameters load bit-exact in CPU runtime without GPU/CUDA. |
| **CAP-02** | **Persona Identity & Self-Introduction** | **PROVEN** | Prompt `"Who are you?"` reliably generates: `"I am ChakrView, an indigenous neural language model developed from first principles."` Zero mode collapse or syntax corruption on identity queries. |
| **CAP-03** | **Turn-Boundary & Generation Termination** | **PROVEN** | Emits clean EOS and conversational turn breaks; eliminates infinite repetition loops on in-distribution dialogues. |
| **CAP-04** | **Held-Out Cross-Entropy Reduction** | **PROVEN** | Test loss dropped from 5.2353 (Parent) to 3.0480 (Candidate) across 54 held-out examples ($p < 10^{-18}$). Candidate improved on 100% of held-out test cases. |
| **CAP-05** | **Anti-Collapse Attractor Resistance** | **PROVEN** | Candidate exhibits 0 occurrences of `'Cube'`, `'Mumbai'`, or format-lock attractors across the 100-case suite. |
| **CAP-06** | **In-Distribution Contextual Retrieval (CBR-05 format)** | **PARTIALLY PROVEN** | Demonstrates retrieval when formatted identically to CBR-05 synthetic pairs; fails on unconstrained conversational phrasing. |
| **CAP-07** | **Out-of-Distribution Entity Extraction** | **UNPROVEN** | On unseen entity prompts (e.g. `"Alice lives in Seattle. Where does Alice live?"`), emits out-of-context tokens (`'Celsia'`). |
| **CAP-08** | **Robust Epistemic Abstention** | **UNPROVEN** | Frequently emits persona strings or irrelevant phrases on novel unanswerable questions instead of consistent refusal. |
| **CAP-09** | **Multi-Turn Context Tracking (>3 turns)** | **UNPROVEN** | In multi-turn history, conversational drift occurs; fails to recall earlier state variables across intervening turns. |
| **CAP-10** | **Arithmetic & Multi-Step Reasoning** | **UNPROVEN** | Emits invalid arithmetic strings on simple addition/subtraction queries; parameter capacity (3.44M) is insufficient for reasoning. |
| **CAP-11** | **Multilingual Generalization** | **UNPROVEN** | Degenerates into corrupted phonetic fragments or English fallback when prompted in Devanagari Hindi or Hinglish. |

---

## 2. Explicit Scientific Guardrails & Prohibitions
1. **Never claim conversational mastery**: The model is an experimental 3.44M cognitive prototype with proven persona preservation and cross-entropy gains, NOT a general-purpose conversational AI.
2. **Never claim reasoning or arithmetic capability**: ChakrMicro v0.1 does not possess arithmetic reasoning circuits.
3. **Never claim open-ended instruction following**: The model follows narrow SFT formats, but degenerates when given unconstrained natural instructions.

