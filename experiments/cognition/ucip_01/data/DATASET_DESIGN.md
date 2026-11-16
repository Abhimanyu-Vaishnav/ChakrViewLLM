# UCIP-01: Dataset Design & Curriculum Engineering Report

**Mission ID:** UCIP-01  
**Project:** ChakrView  
**Date:** 2026-11-16  
**Auditor / Lead Engineer:** Autonomous Lead Systems Architect  

---

## 1. Executive Summary

Phase D designed and compiled the **Unified Conversational Dataset** (`experiments/cognition/ucip_01/data/ucip01_train.jsonl` and `ucip01_validation.jsonl`) and the **Sealed 1,850-Probe Benchmark** (`experiments/cognition/ucip_01/evaluation/`).

The dataset moves beyond narrow single-turn templates to provide multi-capability exposure:
- **Natural Dialogue & Conversational Continuity:** English, Devanagari Hindi, Roman Hindi, and Hinglish.
- **Memory-Conditioned Dialogue:** Dialogues prefixed with structured memory contexts (`[Memory: Predicate: Value]`) teaching the neural transformer how to incorporate retrieved facts into natural responses.
- **Dynamic State Updates & Corrections:** User renaming projects, correcting personal details, and resolving contradictory claims.
- **Multi-Step Reasoning & Arithmetic:** Basic logical deduction and integer calculations.
- **Uncertainty & Calibrated Abstention:** Recognising missing or unknowable information.

---

## 2. Dataset Counts & Integrity

| Dataset Component | Dialogues / Turns | File Path | Format |
| :--- | :---: | :--- | :--- |
| **UCIP-01 Training Corpus** | 2,142 dialogues (8,568 turns) | `experiments/cognition/ucip_01/data/ucip01_train.jsonl` | JSONL (UTF-8) |
| **UCIP-01 Validation Corpus** | 238 dialogues (952 turns) | `experiments/cognition/ucip_01/data/ucip01_validation.jsonl` | JSONL (UTF-8) |
| **Sealed Independent Benchmark** | 1,850 probes (13 domains) | `experiments/cognition/ucip_01/evaluation/*.json` | JSON |

---

## 3. Sealed Benchmark Domains (1,850 Total Probes)

1. `english_conversation_150.json`: 150 probes
2. `devanagari_hindi_150.json`: 150 probes
3. `roman_hindi_150.json`: 150 probes
4. `hinglish_150.json`: 150 probes
5. `cross_language_100.json`: 100 probes
6. `multiturn_context_200.json`: 200 probes
7. `hybrid_memory_retrieval_200.json`: 200 probes
8. `neural_only_entity_recall_100.json`: 100 probes
9. `user_corrections_150.json`: 150 probes
10. `instruction_following_150.json`: 150 probes
11. `basic_reasoning_150.json`: 150 probes
12. `uncertainty_abstention_100.json`: 100 probes
13. `long_conversation_stability_100.json`: 100 probes
