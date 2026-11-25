# CHAKRVIEW RESEARCH REPORT: NR-013 BASELINE EVALUATION
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Baseline Checkpoint:** `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`  
**Benchmark Suite:** 210 Fresh Sealed Tasks (`data/sealed_eval_013/sealed_benchmark_210.json`)  
**Evaluator:** `experiments/baseline_eval_013/run_baseline_eval.py`  

---

## 1. Executive Summary & Epistemic Honesty
Under the hardened, strict 210-task sealed benchmark, the untouched NR-011 Stage-D foundation model was independently evaluated across both Pure Neural and Hybrid Cognitive execution modes:

- **Pure Neural Baseline:**
  - **Overall Strict Accuracy:** **5.7%** (12/210)
  - **Python Code Execution:** **0.0%** (0/30) — Syntax validity 0.0%
  - **Arithmetic:** **5.0%** (1/20)
  - **Overall Coherence:** **87.1%**
  - **Repetition Rate:** **0.0%**
  - **Average Latency:** 407.5 ms | Throughput: 81.9 tok/s

- **Hybrid Cognitive Baseline:**
  - **Overall Strict Accuracy:** **18.6%** (39/210)
  - **Arithmetic (Deterministic Router):** **100.0%** (20/20)
  - **Safety Abstention:** **45.0%** (9/20)
  - **Multi-Turn Context Recall:** **40.0%** (6/15)
  - **Overall Coherence:** **95.2%**
  - **Average Latency:** 355.4 ms | Throughput: 84.8 tok/s

---

## 2. Forensic Gap Analysis
1. **Python Generation Deficit:** The original Stage-D checkpoint contains zero dedicated code execution conditioning, outputting unstructured text or conversational snippets rather than compilable Python function blocks (`0.0% syntax validity`).
2. **Strict vs Lenient Evaluation:** Unlike NR-012 which counted generic 4-word fluency toward conversation accuracy, the NR-013 hardened evaluator enforces exact keyword or semantic matching.
3. **Target for Phase 5:** Pure neural coding SFT must condition the 17M architecture to output valid Markdown code blocks with syntax validity $\ge 50\%$ and positive assertion execution passes.
