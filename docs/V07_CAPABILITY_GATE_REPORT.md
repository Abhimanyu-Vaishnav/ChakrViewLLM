# ChakrView v0.7 — Capability Gate, Code Integrity & GGUF Proof Report

**Milestone**: `v0.7` Capability Track & GGUF Packaging Proof  
**Target Release**: November 2026 v1.0 Coding-Specialized Model  
**Verdict**: **`V0.7 EXPERIMENT READY — AWAITING TRAINING AUTHORIZATION`**

---

## 1. Executive Summary

This milestone achieved four decisive engineering objectives advancing ChakrView toward the November 2026 v1.0 coding-specialized model:
1. **Hardened Coding Evaluation Suite with Process Isolation**: Upgraded code execution from in-process restricted built-ins to a **process-isolated subprocess runner** with AST static safety auditing, resource bounding, and strict timeout enforcement (tested against infinite loops and prohibited OS imports).
2. **GGUF Serialization Proof**: Developed an indigenous, dependency-free binary GGUF v3 writer ([`chakrview/export/gguf_writer.py`](file:///d:/Project/ChakrView/chakrview/export/gguf_writer.py)) and successfully serialized the canonical Step-10,000 checkpoint into `artifacts/gguf_prototype/chakrmicro_step10000.gguf` (16.20 MB, 57 tensors, 16 metadata KV pairs, 32-byte alignment).
3. **Epistemically Corrected Capability Matrix**: Audited and reconciled capability claims in [`docs/CAPABILITY_MATRIX.md`](file:///d:/Project/ChakrView/docs/CAPABILITY_MATRIX.md), clearly separating pure neural performance from governed retrieval tooling and correcting earlier statements regarding code attractor elimination.
4. **Decision-Ready Single-Field Copy Training Proposal**: Formulated a tightly bounded, leakage-safe training curriculum proposal ([`docs/SINGLE_FIELD_COPY_PILOT_PROPOSAL.md`](file:///d:/Project/ChakrView/docs/SINGLE_FIELD_COPY_PILOT_PROPOSAL.md)) with explicit quantitative stopping gates. **Zero training was executed.**

---

## 2. Repository & Safety Invariant Audits

- **Canonical Checkpoint Hash**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified bit-identical)
- **Canonical Vocab Hash**: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Verified bit-identical)
- **Canonical Merges Hash**: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Verified bit-identical)
- **Protected File**: [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) verified **completely unmodified** (`git diff` clean).
- **Test Set Protection**: Held-out `test.jsonl` was never accessed or inspected.

---

## 3. Hardened Coding Evaluation Suite (Process-Isolated)

### Execution Architecture & Security
- **Static AST Safety Gate**: Scans code AST before execution, rejecting dangerous imports (`os`, `subprocess`, `sys`, `socket`, `requests`, `ctypes`) and dangerous calls (`open`, `eval`, `exec`).
- **Subprocess Isolation**: Executes code in an ephemeral temporary directory with a separate Python interpreter instance.
- **Enforced Timeouts**: Hard termination after 2.0 seconds for runaway loops.

### Step-10,000 Baseline Verification (6 Tasks)
- **Syntax Validity Rate**: **2/6 (33.3%)**
- **Test Pass Rate**: **1/6 (16.7%)**
- **Timeout Rate**: **0/6 (0.0%)**
- **Security Policy Rejection Rate**: **0/6 (0.0%)**
- **Test Suite**: Automated tests in [`tests/test_coding_suite.py`](file:///d:/Project/ChakrView/tests/test_coding_suite.py) pass cleanly (4/4).

---

## 4. GGUF Binary Proof & Runtime Path

### Export Verification
- Implemented [`chakrview/export/gguf_writer.py`](file:///d:/Project/ChakrView/chakrview/export/gguf_writer.py) and [`chakrview/export/export_chakrmicro_gguf.py`](file:///d:/Project/ChakrView/chakrview/export/export_chakrmicro_gguf.py).
- Successfully converted canonical Step-10,000 weights to `artifacts/gguf_prototype/chakrmicro_step10000.gguf`.
- Verified binary header: Magic `b"GGUF"`, Version 3, 57 tensors, 16 metadata KV pairs, exact 32-byte alignment.

### Runtime Integration Plan for v0.8
- Standard 7B/13B models use head dimensions of 64 or 128; ChakrMicro uses $d_{head} = 32$ ($192 / 6$).
- In milestone **v0.8**, we will test generic CPU execution paths in `llama.cpp` and evaluate standard 4-bit (`q4_k_m`) and 8-bit quantization.

---

## 5. Single-Field Copy Pilot: Readiness & Next Gate

Detailed in [`docs/SINGLE_FIELD_COPY_PILOT_PROPOSAL.md`](file:///d:/Project/ChakrView/docs/SINGLE_FIELD_COPY_PILOT_PROPOSAL.md):
- **Hypothesis**: Training ChakrMicro exclusively on single-field entity extraction using dialog framing (`User: ... Assistant: <val>`) will induce context-to-output pointer circuits without overwhelming the model with multi-field JSON schema syntax.
- **Budget**: 250–300 steps on CPU (~45–60 seconds).
- **Predefined Stopping Criteria**: If exact match copying is `< 10.0%` by step 200, terminate immediately and roll back.
- **Status**: **Fully specified and prepared; awaiting explicit user authorization.**

---

## 6. Full Test Suite Execution Summary

```powershell
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```
**Results**: **106 tests executed, 106 passed (100% pass rate, 0 failures, 0 errors)**.

---

## 7. November v1.0 Release Blockers & Highest-Value Next Action

### November v1.0 Blockers:
1. **Zero Neural Context Copying**: The model must learn to copy variables/names from prompt context into output.
2. **Third-Party GGUF Runtime Verification**: Quantized GGUF inference in `llama.cpp` must be benchmarked on $d_{head}=32$.
3. **Public Model Naming & Hugging Face Repository Clearance**.

### Single Highest-Value Next Action:
**Authorize and execute the Single-Field Entity Copy Training Pilot (250–300 steps, CPU).**  
This is the single technical prerequisite gating coding capability. All dataset splits, supervision masks, stopping criteria, and evaluation harnesses are prepared and verified.
