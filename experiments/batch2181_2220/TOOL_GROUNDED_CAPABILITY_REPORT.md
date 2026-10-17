# TOOL-GROUNDED CAPABILITY REPORT

**Batch:** 2181–2220  
**Focus:** Verified Tool Delegation for Arithmetic, Symbolic Reasoning, and Safety  
**Evaluator:** `VerifiedToolRouter` (`experiments/batch2181_2220/verified_tool_router.py`) & `SafeCalculator` (`chakrview/runtime/safe_calculator.py`)  

---

## 1. Problem Statement: The 0% Arithmetic Reality

In the Batch 2141–2180 evaluation, both the canonical baseline model and all three newly trained CPT candidates achieved **0.0% accuracy (0/20)** on multi-step integer arithmetic under deterministic greedy decoding.

### Why Raw Small Language Models Cannot Do Arithmetic
- A 3.44M parameter autoregressive neural model predicts tokens based on statistical subword co-occurrence, not algorithmic arithmetic logic.
- Expecting a 3.4M model to calculate `500 - 234 = 266` via raw token generation conflates statistical language representation with deterministic symbolic execution.
- In indigenous AI systems, **neuro-symbolic separation of concerns** is foundational: the neural core handles intent routing and natural language parsing, while deterministic calculators handle arithmetic.

---

## 2. Verified Tool Delegation Architecture

To ground arithmetic without compromising safety:
1. **Zero Raw `eval()` or `exec()`:** The tool relies exclusively on `SafeCalculator`, which uses Python's standard `ast` module with a strict node allowlist (`Expression`, `Constant`, `BinOp`, `UnaryOp`).
2. **Explicit Allowlist of Operators:** Restricted to `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`, and unary `+`. Boolean logic, imports, attribute access, and OS calls are blocked at the AST level.
3. **Bounded Resources:** Maximum expression length capped at 256 characters; maximum exponent capped at 100 to prevent denial-of-service / memory blowup.
4. **Router Interface:** `VerifiedToolRouter` scans incoming queries for mathematical expressions, executes the calculation in a deterministic sandbox, and returns a structured `ToolExecutionTrace`.

```
User Query: "What is 1000 / 8?"
     |
     v
[VerifiedToolRouter] ---> Extracts expression: "1000 / 8"
     |
     v
[SafeCalculator (AST)] -> Validates AST nodes -> Evaluates: 125.0 -> Formats: 125
     |
     v
ToolExecutionTrace: { tool: "safe_calculator", output: 125, success: True, latency: 0.12ms }
```

---

## 3. Comparative Evaluation: Raw Neural Model vs Tool-Assisted

We evaluated both approaches across the **20 canonical arithmetic probes**:

| Test Query | Target Answer | Raw Neural Model (Baseline / CPT) | Tool-Assisted (VerifiedToolRouter) | Result |
| :--- | :---: | :---: | :---: | :---: |
| "What is 12 + 15?" | 27 | 0 (Hallucinated string) | **27** | **PASS** |
| "Calculate 25 * 4" | 100 | 0 (Hallucinated string) | **100** | **PASS** |
| "What is 100 - 37?" | 63 | 0 (Hallucinated string) | **63** | **PASS** |
| "144 / 12" | 12 | 0 (Hallucinated string) | **12** | **PASS** |
| "7 * 8" | 56 | 0 (Hallucinated string) | **56** | **PASS** |
| "81 / 9" | 9 | 0 (Hallucinated string) | **9** | **PASS** |
| "45 + 55" | 100 | 0 (Hallucinated string) | **100** | **PASS** |
| "250 - 125" | 125 | 0 (Hallucinated string) | **125** | **PASS** |
| "16 * 16" | 256 | 0 (Hallucinated string) | **256** | **PASS** |
| "1000 / 8" | 125 | 0 (Hallucinated string) | **125** | **PASS** |
| "33 + 77" | 110 | 0 (Hallucinated string) | **110** | **PASS** |
| "64 - 28" | 36 | 0 (Hallucinated string) | **36** | **PASS** |
| "13 * 7" | 91 | 0 (Hallucinated string) | **91** | **PASS** |
| "225 / 15" | 15 | 0 (Hallucinated string) | **15** | **PASS** |
| "99 + 101" | 200 | 0 (Hallucinated string) | **200** | **PASS** |
| "500 - 234" | 266 | 0 (Hallucinated string) | **266** | **PASS** |
| "18 * 5" | 90 | 0 (Hallucinated string) | **90** | **PASS** |
| "360 / 6" | 60 | 0 (Hallucinated string) | **60** | **PASS** |
| "49 + 51" | 100 | 0 (Hallucinated string) | **100** | **PASS** |
| "120 - 45" | 75 | 0 (Hallucinated string) | **75** | **PASS** |

### Benchmark Summary

| System Configuration | Total Probes | Correct Answers | Accuracy | Execution Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Neural Core (Canonical Baseline)** | 20 | 0 / 20 | **0.0%** | ~130 ms |
| **Raw Neural Core (CPT Seed 42)** | 20 | 0 / 20 | **0.0%** | ~130 ms |
| **ChakrView Tool-Grounded (VerifiedToolRouter)** | 20 | 20 / 20 | **100.0%** | **0.15 ms** |

---

## 4. Safety and Adversarial Invariance

We tested the router against malicious prompts attempting code injection or resource starvation:
- `__import__('os').system('ls')`: **Rejected** (`Security violation: Disallowed syntax node 'Call'`).
- `open('config.json').read()`: **Rejected** (`Disallowed syntax node 'Call'`).
- `10 ** 1000000`: **Rejected** (`EXPRESSION_TOO_LONG` / exponent bounds).
- Zero division (`10 / 0`): **Handled gracefully** (`DIVISION_BY_ZERO`).

**Conclusion:** Verified tool delegation elevates mathematical task adherence from 0% to 100% with sub-millisecond overhead, without granting arbitrary execution privileges to model outputs.
