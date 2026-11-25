# NR-015 Phase 7 — Neural Coding Generalization Report
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Model:** `ChakrGenesis-17M Stage-D Best` (`2c21632e`)  
**Test Suite:** 40 unseen Python coding tasks with isolated subprocess execution and hidden unit test assertions  
**Artifacts:** `data/sealed_eval_015/coding_eval_results.json`

---

## 1. Quantitative Benchmark Results

| Coding Metric | Measured Performance | Rate |
| :--- | :---: | :---: |
| **Total Evaluated Tasks** | 40 | $100\%$ |
| **AST Syntax Validity (`ast.parse`)** | 28 / 40 | **$70.0\%$** |
| **Executable Assertion Pass Rate** | 0 / 40 | **$0.0\%$** |
| **Dominant Failure Mode** | Target function undefined / hallucinated comments | NameError / SyntaxError |

---

## 2. Mechanistic Failure Modes
Inspection of generated tokens isolates three primary behaviors:
1. **Comment / Docstring Hallucination:** Rather than generating the requested function definition (`def square_number(x):`), the model outputs comments memorized from pretraining files (`# Information the context`, `# Initialize the context`).
2. **Import Loop Hallucination:** The model initiates import statements (`from dataclass`, `import codecs`) followed by repetitive variable names (`isinstance(path, path, path, path...)`), indicating an attention attractor state.
3. **Absence of Signature Grounding:** While high-level syntax is recognizable (yielding a 70.0% validity rate on simple statements), the network fails to bind the requested function name from the prompt to the `def` statement header.
