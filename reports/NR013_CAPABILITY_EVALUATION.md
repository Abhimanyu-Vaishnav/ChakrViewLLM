# CHAKRVIEW RESEARCH REPORT: NR-013 CAPABILITY EVALUATION
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Evaluation Target:** 210 Sealed Hardened Tasks across 10 Categories  
**Execution Environment:** Isolated Python subprocess execution (2.0s hard timeout)  

---

## 1. Executive Summary & Epistemic Honesty
Under Mission `CV-GENESIS-NR-013`, we evaluated the baseline `ChakrGenesis-17M Stage-D` model and the newly trained `ChakrGenesis-17M Stage-E` candidate on an independent 210-task benchmark.

### Primary Capability Findings:
1. **Python Syntax Validity Emergence:**
   - Baseline Stage-D produced **0.0%** compilable Python code (0/30 syntax valid).
   - Candidate Stage-E successfully synthesized compilable Python syntax, increasing syntax validity to **20.0%** (6/30 compilable functions).
2. **Execution Correctness Constraint:**
   - Both models scored **0.0%** on hidden assertion execution passes (0/30). At the 17M parameter scale, zero-shot code synthesis frequently hallucinates function parameters or completes with generic functions from memory rather than strictly adhering to novel query signatures.
3. **Hybrid Cognitive Verification:**
   - The Hybrid Cognitive System achieves **18.6% overall strict accuracy** (39/210), driven by 100.0% deterministic arithmetic (20/20), 45.0% safety abstention (9/20), and 40.0% multi-turn memory recall (6/15).

---

## 2. Multi-Model Capability Matrix (210 Tasks)

| Category | Item Count | Baseline 17M Stage-D (Pure) | Candidate 17M Stage-E (Pure) | Hybrid Cognitive (17M Stage-E) |
| :--- | :--- | :--- | :--- | :--- |
| **Python Code Generation** | 30 | 0.0% (0% syntax) | 0.0% (**20.0% syntax**) | 0.0% (**20.0% syntax**) |
| **Python Explanation** | 15 | 0.0% | 0.0% | 0.0% |
| **Algorithmic Reasoning** | 15 | 0.0% | 0.0% | 0.0% |
| **Arithmetic** | 20 | 5.0% (1/20) | 5.0% (1/20) | **100.0%** (20/20) |
| **English Conversation** | 25 | 8.0% (2/25) | 0.0% (0/25) | 4.0% (1/25) |
| **Hindi Conversation** | 25 | 0.0% (0/25) | 0.0% (0/25) | 4.0% (1/25) |
| **Hinglish Dialogue** | 25 | 20.0% (5/25) | 8.0% (2/25) | 8.0% (2/25) |
| **Instruction Following** | 20 | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| **Multi-Turn Context** | 15 | 0.0% (0/15) | 0.0% (0/15) | **40.0%** (6/15) |
| **Safety Abstention** | 20 | 20.0% (4/20) | 10.0% (2/20) | **45.0%** (9/20) |
| **Overall Strict Accuracy**| **210** | **5.7%** (12/210) | **2.4%** (5/210) | **18.6%** (39/210) |
| **Overall Coherence Rate** | **210** | **87.1%** | **86.7%** | **95.7%** |
| **Repetition / Degeneracy** | **210** | **0.0%** | **0.0%** | **0.0%** |
| **Throughput (tokens/sec)** | — | **81.9 tok/s** | **72.4 tok/s** | **82.3 tok/s** |
