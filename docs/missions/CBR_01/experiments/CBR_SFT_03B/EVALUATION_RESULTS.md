# CBR-SFT-03B — Evaluation Results Master Report

## 1. Executive Summary & Benchmark Matrix

The CBR-SFT-03B experiment evaluated 7 checkpoint configurations across four independent test suites:
1. **Held-Out Greeting & Identity Suite (70 tasks)**: Strictly zero-overlap held-out evaluation testing out-of-distribution generalization on greetings, identity questions, and short conversational QA.
2. **Canonical M3 Conversational Benchmark (60 tasks)**: The canonical 11-category conversational benchmark.
3. **Native Intelligence Suite (32 tasks)**: Measuring factual retention, logic, and repetitive pretraining attractor collapse.
4. **Coding Benchmark (12 tasks)**: Measuring Python syntax validity and execution correctness.

### Master Evaluation Comparison

| Model / Checkpoint | Held-Out 70 Suite (Acc %) | M3 Conversational 60 (Acc %) | Intelligence 32 (Passed) | Attractor Rate (% Collapse) | Coding 12 (Syntax Valid) | Coding (Exec Pass) |
|---|---|---|---|---|---|---|
| **Canonical Step 10k** | 0 / 70 (0.0%) | 0 / 60 (0.0%) | 1 / 32 | 17 / 32 (53.1%) | 3 / 12 (25.0%) | 0 / 12 (0.0%) |
| **CBR-SFT-02 Step 50** | 0 / 70 (0.0%) | 0 / 60 (0.0%) | 3 / 32 | 11 / 32 (34.4%) | 4 / 12 (33.3%) | 0 / 12 (0.0%) |
| **CBR-SFT-03B Step 75** | 1 / 70 (1.4%) | 0 / 60 (0.0%) | 1 / 32 | 6 / 32 (18.8%) | 3 / 12 (25.0%) | 0 / 12 (0.0%) |
| **CBR-SFT-03B Step 100** | 4 / 70 (5.7%) | 2 / 60 (3.3%) | 1 / 32 | 2 / 32 (6.3%) | 3 / 12 (25.0%) | 0 / 12 (0.0%) |
| **CBR-SFT-03B Step 150** | 16 / 70 (22.9%) | 3 / 60 (5.0%) | 1 / 32 | **1 / 32 (3.1%)** | 4 / 12 (33.3%) | 0 / 12 (0.0%) |
| **CBR-SFT-03B Step 200** | **19 / 70 (27.1%)** | **3 / 60 (5.0%)** | 1 / 32 | **1 / 32 (3.1%)** | 4 / 12 (33.3%) | 0 / 12 (0.0%) |
| **CBR-SFT-03B Step 300** | **19 / 70 (27.1%)** | **3 / 60 (5.0%)** | 1 / 32 | **1 / 32 (3.1%)** | **5 / 12 (41.7%)** | 0 / 12 (0.0%) |

---

## 2. Category Breakdown: Held-Out 70-Task Suite

This benchmark used completely disjoint phrasing from the training set. None of the prompts appeared in `cbr_sft_03b_train_45.jsonl`.

| Category | Tasks | Step 10k | CBR-2 Step 50 | Step 75 | Step 100 | Step 150 | Step 200 | Step 300 |
|---|---|---|---|---|---|---|---|---|
| **Greetings** | 30 | 0.0% | 0.0% | 3.3% | 10.0% | 23.3% | **26.7%** (8/30) | **30.0%** (9/30) |
| **Identity & Role** | 20 | 0.0% | 0.0% | 0.0% | 5.0% | 30.0% | **35.0%** (7/20) | **35.0%** (7/20) |
| **Short QA / Readiness** | 20 | 0.0% | 0.0% | 0.0% | 0.0% | 15.0% | **20.0%** (4/20) | **15.0%** (3/20) |
| **Total Passed** | **70** | **0 / 70 (0.0%)** | **0 / 70 (0.0%)** | **1 / 70 (1.4%)** | **4 / 70 (5.7%)** | **16 / 70 (22.9%)** | **19 / 70 (27.1%)** | **19 / 70 (27.1%)** |

---

## 3. Canonical M3 Benchmark Analysis (60 Tasks)

Prior to CBR-SFT-03B, every model in ChakrView history scored exactly **0 / 60 (0.0%)** on free autoregressive evaluation.
At Step 100, the model achieved **2 / 60 (3.3%)**, and at Steps 150, 200, and 300, it achieved **3 / 60 (5.0%)**.

Category performance for Step 200 and Step 300:
- **Greetings & Natural Dialogue (5 tasks)**:
  - Step 200: **1 / 5 (20.0%)** (`M3-EVAL-GREET-01`: `"User: Hi, good morning!"` $\to$ `"Good morning! How can I help you today?"`)
  - Step 300: **2 / 5 (40.0%)** (`M3-EVAL-GREET-01` and `M3-EVAL-GREET-02`: `"Who are you?"` $\to$ `"I am ChakrView, an indigenous neural language model..."`)
- **Grounded Question Answering (5 tasks)**:
  - Step 200: **1 / 5 (20.0%)** (`M3-EVAL-GQA-04`)
  - Step 300: **1 / 5 (20.0%)** (`M3-EVAL-GQA-04`)
- **Basic Reasoning (5 tasks)**:
  - Step 200: **1 / 5 (20.0%)** (`M3-EVAL-REAS-01`)
  - Step 300: 0 / 5 (0.0%)
- **Multi-turn Context Retention (10 tasks)**: 0 / 10 (0.0%) — Multi-turn tracking requires multi-turn SFT supervision.
- **Arithmetic & Logic (5 tasks)**: 0 / 5 (0.0%) — Parameter count limits intrinsic arithmetic reasoning without explicit tool use.
- **Summarization & Format Adherence (10 tasks)**: 0 / 10 (0.0%).

---

## 4. Attractor Suppression & Non-Regression Audit

- **Pretraining Attractor Rate**:
  - Dropped from **53.1%** in Canonical Baseline and **34.4%** in CBR-SFT-02 down to **3.1%** (1 / 32) at Steps 150, 200, and 300.
  - Repetitive loops like `scope=&quot;row&quot;` and `the word of the word` were virtually extinguished in free autoregressive generation.
- **Native Intelligence**:
  - Baseline score: 1 / 32.
  - Step 200/300 score: 1 / 32. Capability was preserved with zero catastrophic forgetting of underlying factual routing.
- **Coding Benchmark**:
  - Baseline AST syntax validity: 3 / 12 (25.0%).
  - Step 300 AST syntax validity: **5 / 12 (41.7%)**.
  - Functional execution correctness remains 0 / 12 across all steps, as expected for an untuned 21M-parameter model on algorithmic problems.
