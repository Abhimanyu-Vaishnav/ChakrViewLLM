# ChakrView v0.7.0 Release Notes — Coding Specialization Foundation

**Version**: `v0.7.0-candidate`  
**Classification**: Architecture Target & Coding Capability Baseline  
**Lineage**: Builds directly upon ChakrView v0.6.0 Research Preview  
**Status**: Experimental Research Baseline  

---

## 1. Executive Summary

ChakrView v0.7.0 establishes the **coding-specialization foundation** for the indigenous 3.44M-parameter neural architecture (`ChakrMicro`).

Rather than overclaiming acquired capabilities, v0.7.0 delivers:
1. **Reproducible 6-Category Coding Evaluation Harness**: Sandboxed, time-limited execution evaluating Python code across 6 core tasks.
2. **Coding Data Pipeline & Leakage Auditor**: Formal schema, duplicate detection, template family overlap auditing, and target leakage verification.
3. **Integrated Coding Assistant Interface**: Extends the existing runtime with code drafting, explanation, bug fixing, test generation, and isolated execution.
4. **Empirical Capability Diagnostics**: Establishes exact baseline performance for the canonical Step 10,000 checkpoint ($\Delta W = 0$).

---

## 2. Key Deliverables & Components

### 2.1 Sandboxed Coding Evaluation Suite (`evaluation/coding_suite/`)
- Evaluates code across 6 task categories: Continuation, Explanation, Bug Repair, Small Implementation, Test Generation, and Summarization.
- Process-isolated subprocess runner with AST safety audit (blocking `os`, `subprocess`, `open`, `eval`, `exec`).
- Automated reporting of syntax validity, unit-test pass rates, and throughput.

### 2.2 Reusable Coding Data Pipeline (`chakrview/data/coding_pipeline.py`)
- Versioned record schema: `task_id`, `task_type`, `language`, `context`, `prompt`, `response`, `provenance`, `split`, `template_family`.
- Validation against malformed records, prompt duplication, and cross-split contamination.
- Deterministic dataset manifests with cryptographic SHA-256 hashes.

### 2.3 Practical Coding Workflow (`chakrview/runtime/coding_assistant.py`)
- Direct integration into `assistant_cli.py`:
  - `python -m chakrview.runtime.assistant_cli code draft "<spec>"`
  - `python -m chakrview.runtime.assistant_cli code explain "<code>"`
  - `python -m chakrview.runtime.assistant_cli code fix "<code>" --error "<msg>"`
  - `python -m chakrview.runtime.assistant_cli code test "<code>"`
  - `python -m chakrview.runtime.assistant_cli code sandbox "<code>" --test-assertions "<tests>"`
- Strictly separates native model completions from tool-assisted execution.

---

## 3. Empirical Baseline Performance (Step 10,000 Canonical Core)

Evaluating the base pretraining model (`ChakrMicro` Step 10,000, 3.44M parameters) on the 6-task benchmark reveals:

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| **Total Tasks Evaluated** | 6 | 6 core software categories covered |
| **Overall Test Pass Rate** | **1/6 (16.7%)** | Simple assert test passes; multi-token algorithms unlearned |
| **Syntax Validity Rate** | **3/6 (50.0%)** | Model generates syntactically valid Python tokens ~50% of the time |
| **Generation Speed (CPU)** | **~138 tok/s** | Extremely fast cold-start and CPU token throughput |
| **Security Policy Violations** | **0** | Clean, safe sandboxed execution |
| **Timeouts** | **0** | All generation and execution completed within 2.0s limit |

---

## 4. Epistemic Transparency & Boundaries

- **Not Fine-Tuned for Coding Yet**: The canonical checkpoint remains the pretraining baseline ($\Delta W = 0$). SFT specialized for code syntax and multi-step logic is scheduled for subsequent milestones (v0.8 $\to$ v1.0).
- **Tool Scaffolding vs Neural Reasoning**: Sandboxed execution and AST parsing are external verification tools, not proof of native neural intelligence.
- **Weight Sovereignty**: The canonical model checkpoint and tokenizer tables remain untouched and bit-identical.
