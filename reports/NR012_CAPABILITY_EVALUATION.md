# CHAKRVIEW RESEARCH REPORT: NR-012 CAPABILITY EVALUATION
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Benchmark Suite:** 150 Sealed Independent Prompts (`data/sealed_eval_012/sealed_benchmark_150.json`)  
**Evaluator:** `experiments/sealed_eval_012/run_sealed_eval_150.py`  

---

## 1. Executive Summary
A fresh 150-prompt sealed evaluation suite was executed across three system configurations:
1. **ChakrGenesis-17M Stage-D (Hybrid Cognitive Mode)**
2. **ChakrGenesis-17M Stage-D (Pure Neural Mode)**
3. **ChakrGenesis-17M NR-010 SFT (Baseline Neural Mode)**

---

## 2. Benchmark Comparison Summary

| Metric / Category | 17M Stage-D (Hybrid) | 17M Stage-D (Pure Neural) | 17M NR-010 SFT (Baseline) |
| :--- | :--- | :--- | :--- |
| **Overall Strict Accuracy** | **80.0%** (120/150) | 56.0% (84/150) | 54.7% (82/150) |
| **Overall Coherence Rate** | **93.3%** | 81.3% | 80.7% |
| **Repetition / Degeneracy Rate** | **0.0%** | 0.0% | 0.0% |
| **Average Latency (per prompt)** | **390.1 ms** | 488.9 ms | 472.3 ms |
| **Inference Throughput** | **67.8 tok/s** | 57.9 tok/s | 56.8 tok/s |
| **English Conversation (30)** | 86.7% (26/30) | **93.3%** (28/30) | 90.0% (27/30) |
| **Hindi Conversation (30)** | 96.7% (29/30) | **100.0%** (30/30) | 100.0% (30/30) |
| **Hinglish Conversation (30)** | **83.3%** (25/30) | 80.0% (24/30) | 83.3% (25/30) |
| **Python Code Execution (20)** | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| **Exact Arithmetic (20)** | **100.0%** (20/20) | 0.0% (0/20) | 0.0% (0/20) |
| **Working Memory Recall (10)** | **100.0%** (10/10) | 0.0% (0/10) | 0.0% (0/10) |
| **Principled Abstention (10)** | **100.0%** (10/10) | 20.0% (2/10) | 0.0% (0/10) |

---

## 3. Forensic Analysis & Epistemic Demarcation
- **Conversational Fluency:** Pure neural 17M achieves 93.3% to 100% coherence and keyword alignment on English and Hindi queries.
- **Arithmetic & Fact Recall:** The pure neural network alone scores 0% on exact multi-digit arithmetic and multi-turn state recall, confirming that autoregressive token sampling alone cannot guarantee symbolic precision.
- **Cognitive Integration:** The Hybrid Cognitive Mode elevates total application task accuracy from 56.0% to **80.0%** while reducing average query latency by 20% due to instant tool resolution.
