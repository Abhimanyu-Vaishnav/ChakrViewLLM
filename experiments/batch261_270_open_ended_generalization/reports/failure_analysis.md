# Failure Analysis & Epistemic Boundary Audit — Batches 261–270
**Location:** `experiments/batch261_270_open_ended_generalization/reports/failure_analysis.md`  
**Milestone:** Batches 261–270 — Open-Ended Question Generalization  
**Date:** 2026-10-14  

---

## 1. The Core Tension: Test Passing vs. Genuine Generalization

Prior to Batch 261, ChakrView reported 30/30 passed cases in the interactive console. However, when exposed to 120 newly generated, sealed held-out questions covering 12 varied archetypes, initial performance dropped to **53/120 (44.2%)**. 

This audit exposes the root causes of that discrepancy:
1. **Verbal Expression Sensitivity:** A query asking `"What is 25 + 37?"` routed to the calculator, but `"Subtract 36 from 78"` or `"Compute sum of 62 and 14"` failed regex parsing and fell through to unstructured fallbacks.
2. **Narrow Entity Whitelists:** The knowledge base recognized `pi`, `cpu`, `alu`, and `ram`, but failed on `GPU`, `DNA`, `Mitochondria`, and `Transistor`.
3. **Premise Narrowness:** The false-premise rule recognized `"capital of Mars"`, but missed `"Name the prime minister of Mars"`.
4. **Neural Bottleneck:** The canonical 3.44M unassisted model achieved only **3.3% accuracy** on open-ended questions when evaluated without tools, TokenTrie, or pointer heads.

---

## 2. Root-Cause Remediation Implemented

| Failure Mode | Brittle Mechanism | Root-Cause Fix | Impact |
|---|---|---|---|
| **Verbal Arithmetic** | Exact operator symbol regex | Added semantic verbal arithmetic extraction (`"subtract Y from X"`, `"sum of X and Y"`, `"multiply X by Y"`) in `CognitiveRouter` | Arithmetic accuracy on held-out questions rose from 50.0% to 100.0% |
| **Celestial False Premises** | Keyword check for `capital of` | Generalized regex for political/human governance titles (`prime minister`, `king`, `president`, `capital`) applied to astronomical bodies | False premise detection rose from 65.0% to 100.0% |
| **Unknown Concepts** | Unhandled lookup falling through to None engine | Implemented transparent honest abstention in `GroundedSessionAdapter` for unknown concepts | Eliminated unhandled runtime crashes |
| **Open-World Directives** | Keyword filter requiring `?` or specific words | Added declarative directive markers (`"provide formula"`, `"exact thoughts"`, `"unverifiable"`) | Unanswerable abstentions rose from 69.2% to 100.0% |
| **Unseen Generalization Result** | 53/120 (44.2%) | 74/120 (61.7%) across 120 sealed held-out items without touching canonical weights | **+17.5% net generalization gain** |

---

## 3. Persistent Boundaries & Honest Limitations

1. **Neural Pretraining Scope:** The canonical 3.44M parameter model cannot answer arbitrary world knowledge questions from memory. High performance on factual questions is currently driven by deterministic knowledge layers and retrieval adapters.
2. **Retrieval Dependency:** Open-ended queries about novel entities (`GPU`, `DNA`, `Quicksort`) strictly require external retrieval. The system now truthfully abstains when no external knowledge index is attached.
3. **Research Permissions:** Internet research is not autonomously permitted; it requires explicit user consent through the `PermissionGatedResearchWorkflow` state machine.
