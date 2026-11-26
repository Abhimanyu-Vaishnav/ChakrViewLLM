# NR-019: Router Architecture Comparison Report

## 1. Five Systems Overview
Mission CV-GENESIS-NR-019 compared five distinct architectural paths on the exact same sealed 800-task benchmark (0 duplicate prompts):
- **System A (Pure Neural Baseline)**: ChakrGenesis-17M Stage-D generating answers autoregressively end-to-end without tools.
- **System B (Deterministic Pattern Parser)**: NR-018 regex template parser dispatched directly to the restricted execution engine.
- **System C (Neural Router Direct)**: 17M foundation model fine-tuned to emit full JSON payloads (`{"operation": "...", "arguments": {...}}`) with free-form token generation.
- **System D (Neural Router + Source-Span Grounding)**: 17M foundation model identifies operation / intent, while arguments are bound directly to exact character source-spans in the user prompt and normalized deterministically.
- **System E (Cascaded Hybrid Router)**: High-speed deterministic regex parser fast-path; falls back to Neural Source-Grounded routing on unmatched or uncertain inputs.

---

## 2. Quantitative Benchmark Results (800 Sealed Tasks)

| System | Overall Accuracy | In-Distribution (300) | Out-of-Distribution (500) | English (200) | Hindi (200) | Hinglish (200) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Pure Neural Baseline** | **2.62%** (21/800) | 2.33% | 2.80% | 4.0% | 5.0% | 1.5% |
| **System B: Deterministic Pattern Parser** | **17.62%** (141/800) | 5.67% | 24.80% | 37.5% | 19.0% | 14.0% |
| **System C: Neural Router Direct** | **7.38%** (59/800) | 3.33% | 9.80% | 10.5% | 10.0% | 9.0% |
| **System D: Neural Source-Grounded** | **50.50%** (404/800) | **38.33%** | **57.80%** | **72.5%** | **60.5%** | **69.0%** |
| **System E: Cascaded Hybrid Router** | **51.50%** (412/800) | **39.67%** | **58.60%** | **76.5%** | **60.5%** | **69.0%** |

---

## 3. Scientific Findings
1. **Exact Source-Span Grounding is the Critical Catalyst**:
   - Free-form JSON generation (System C) achieved only 7.38% accuracy on the diverse 800-task benchmark due to numerical token hallucination.
   - Binding arguments directly to original source text (System D) boosted accuracy to **50.50%**—a **$6.8\times$ improvement over System C** and a **$19.3\times$ improvement over pure neural inference**.
2. **Deterministic Cascading Provides Modest Additional Uplift**:
   - System E (Parser $\to$ Grounded Router Fallback) achieved the highest overall score of **51.50%** (412/800).
   - Fast regexes provide immediate matching for canonical English arithmetic expressions, while the neural router handles flexible phrasing, Hindi Devanagari numerals, and Hinglish.
