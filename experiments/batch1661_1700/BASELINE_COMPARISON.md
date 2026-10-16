# BASELINE COMPARISON & MULTI-SYSTEM ROUTING BENCHMARK
## Batches 1661–1700 Empirical Comparison Across 120 Frozen Held-Out Tasks

### 1. Benchmark Overview & Setup

To objectively determine whether ChakrView's neural core can perform semantic intent classification and argument extraction, we evaluated five distinct routing architectures against the exact same frozen, leakage-controlled held-out test suite (120 tasks, 20 per intent class, SHA-256: `6d56d955...`):

1. **Deterministic Governed Router (`GovernedToolRouter`):** The existing production router using strict regex patterns and verification state.
2. **Keyword Heuristic Baseline:** A simple lexical substring matcher.
3. **Untrained Canonical ChakrMicro:** Autoregressive prompting on base step-10,000 weights without router adaptation.
4. **Trained Candidate ChakrMicro (Step 100):** Causal LM fine-tuned for 100 steps on structured JSON intent targets.
5. **Governed Neuro-Symbolic Hybrid:** Attempts neural proposal first; verifies schema compliance; falls back to deterministic router on malformed/unsupported output.

---

### 2. Multi-System Performance Matrix

| Metric | Governed Deterministic | Keyword Heuristic | Canonical Neural (Untrained) | Candidate Neural (Trained Step 100) | Governed Neuro-Symbolic Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intent Accuracy** | 23.33% (28/120) | **57.50%** (69/120) | 16.67% (20/120) | 19.17% (23/120) | 23.33% (28/120) |
| **Structured Argument Match** | 2.50% (3/120) | **50.00%** (60/120) | 16.67% (20/120) | 14.17% (17/120) | 2.50% (3/120) |
| **Macro-F1 (6 Intents)** | 0.1585 | **0.5365** | 0.0476 | 0.1001 | 0.1585 |
| **JSON Validity Rate** | **100.0%** | **100.0%** | 0.0% | 14.17% | **100.0%** |
| **Average Latency (CPU)** | **0.06 ms** | **<0.01 ms** | 188.64 ms | 259.08 ms | 289.17 ms |
| **Attribution Breakdown** | 120 Deterministic | 120 Keyword | 120 Neural | 120 Neural | 103 Det. / 17 Neural |

---

### 3. Per-Class Precision & Recall Comparison

#### 3.1 Keyword Heuristic Baseline (Macro-F1: 0.5365)
- **CALCULATE:** Precision 66.7%, Recall 70.0%, F1 0.6829
- **VERIFY_CLAIM:** Precision 68.4%, Recall 65.0%, F1 0.6667
- **KNOWLEDGE_QUERY:** Precision 68.4%, Recall 65.0%, F1 0.6667
- **CODE_TASK:** Precision 72.7%, Recall 80.0%, F1 0.7619
- **CLARIFY:** Precision 0.0%, Recall 0.0%, F1 0.0000 (Cannot detect missing arguments)
- **OUT_OF_SCOPE:** Precision 33.3%, Recall 65.0%, F1 0.4407 (Vulnerable to keyword traps)

#### 3.2 Candidate Neural Router (Step 100) (Macro-F1: 0.1001)
- **CALCULATE:** 0/20 (Fails to format JSON calculation schema under unseen phrasing)
- **VERIFY_CLAIM:** 0/20 (Outputs garbled JSON syntax or drifts into text)
- **KNOWLEDGE_QUERY:** 0/20
- **CODE_TASK:** 0/20
- **CLARIFY:** Precision 35.3%, Recall 30.0%, F1 0.3243 (Correctly learned some missing-argument triggers)
- **OUT_OF_SCOPE:** Precision 16.5%, Recall 85.0%, F1 0.2764 (Default fallback bias)
- **JSON Syntax Failure Rate:** **85.83%** of generations were non-JSON or malformed.

#### 3.3 Governed Deterministic Router (Macro-F1: 0.1585)
- Highly reliable and safe when patterns match (100% precision on CALCULATE when triggered).
- Fails on paraphrases like `"I need to find the sum of 512 and 1024"` because regexes require `"calculate:"` or `"what is"`.
- Safely abstains on 100% of CLARIFY tasks, avoiding accidental false executions.

---

### 4. Empirical Verdict
The candidate neural router **fails to outperform simpler baselines on held-out inputs**.
- It is ~4,000x slower than the deterministic baseline (259 ms vs 0.06 ms).
- It produces invalid JSON in 85.8% of cases.
- Its intent accuracy (19.17%) is barely above random baseline (16.67%) and vastly inferior to simple keyword matching (57.50%).
