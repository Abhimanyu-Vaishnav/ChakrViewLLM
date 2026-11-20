# UCIP-14: Evaluation Protocol & Benchmark Architecture
## Strict Capability Measurement, Evaluator Rigor & Benchmark Isolation

**Document ID:** UCIP-14-EVAL-01  
**Project:** ChakrView  
**Status:** DESIGN ONLY — TRAINING NOT AUTHORIZED  

---

## 1. Remediation of Historical Evaluator Vulnerabilities

Historical audit missions (UCIP-10E-R1, UCIP-10E-R2, and UCIP-11) revealed severe methodological vulnerabilities in earlier evaluation scripts:
- **Substring False Positives:** Evaluators accepted cyclic word salad because a target keyword appeared somewhere in an ungrammatical loop.
- **Length Biases:** Long outputs were scored as passes regardless of correctness.
- **Prompt Duplication:** Repeating identical prompts with minor variations inflated nominal probe counts.

### Non-Negotiable Evaluator Rules for UCIP-14:
1. **No Keyword-Only Scoring:** An answer containing the keyword within a repetitive sequence (e.g. `Paris Paris Paris Paris`) is scored as **FAILED**.
2. **Repetition Penalty Gate:** Any generation containing an identical 3-gram repeated $\ge 3$ times is marked as degenerate repetition and assigned a score of **0.0**.
3. **EOS Enforcement:** An assistant response that fails to emit `<EOS>=1` before reaching the maximum token cutoff receives an **EOS Failure penalty**.
4. **Isolated Subprocess Code Execution:** Python code evaluations are executed in an isolated subprocess with a strict 3.0-second timeout, restricted memory, and zero network access. Untrusted code is never executed directly in the main audit process.
5. **Exact Match & Structured Parsing:**
   - Arithmetic answers require exact numerical evaluation matching ground truth.
   - JSON formatting probes require parsing by `json.loads`.

---

## 2. Capability Evaluation Dimensions

| Capability Dimension | Measurement Methodology | Acceptance Criteria (Provisional) | Baseline (Step 800) |
| :--- | :--- | :--- | :--- |
| **Instruction Adherence** | Exact constraint validation (length, formatting, bullet points) | $\ge 60\%$ adherence on clean unseen probes | ~0% (Document continuation) |
| **Conversational Coherence** | Human-audit grading on relevance, appropriateness, politeness | $\ge 65\%$ coherent responses | ~15% (Disjoint phrases) |
| **Context Retention (Multi-Turn)** | Exact entity/attribute recall across 2-turn and 3-turn dialogues | $\ge 50\%$ correct retention | ~0% (Context loss) |
| **Exact Copying** | Exact string reproduction (tokens, hashes, identifiers) | $\ge 75\%$ character-level match | ~20% (Scrambled tokens) |
| **Entity Binding** | Attribute-to-entity query correctness | $\ge 55\%$ correct binding | ~10% (Swapped entities) |
| **Arithmetic** | Exact numerical equality on word problems and equations | $\ge 50\%$ exact match | ~5% (Random numbers) |
| **Python Execution** | Subprocess test pass rate on simple algorithmic functions | $\ge 40\%$ tests passing | ~0% (Broken syntax loops) |
| **Multilingual Balance** | Separate evaluation on English, Hindi, Hinglish, and Sanskrit | All languages achieve $>40\%$ validity | English ~30%, Hindi <10% |
| **Uncertainty / Abstention** | Appropriate refusal on unanswerable/missing information probes | $\ge 70\%$ correct abstentions | ~0% (Hallucination) |
| **Repetition Control** | 3-gram repetition frequency across generations | $< 5.0\%$ repetitive outputs | ~45% (under greedy decoding) |
| **EOS Termination** | Emission of `<EOS>=1` within token budget | $> 85\%$ clean terminations | ~5% (Infinite continuation) |

---

## 3. Benchmark Partitioning & Isolation Protocol

To guarantee scientific validity and eliminate data contamination, four strictly quarantined evaluation suites are established:

```mermaid
flowchart TD
    A["Master Instruction Data Pool"] --> B["SFT Training Partition (80%)"]
    A --> C["SFT Validation Partition (10%)"]
    A --> D["Dev Evaluation Suite (10%)"]
    
    E["Isolated Quarantine"] --> F["Sealed Final Benchmark (500 Probes)"]
    G["Pretraining Storage"] --> H["Foundational Retention Benchmark (Stage C Test)"]
    
    B -.->|TRAINING ONLY| M["Model Optimizer"]
    C -.->|Hyperparameter Selection| M
    D -.->|Checkpoint Diagnostics| M
    
    F -.->|EVALUATION ONLY (Post-Training)| EV["Final Scientific Audit"]
    H -.->|EVALUATION ONLY (Retention Check)| EV
```

### Strict Isolation Rules:
1. **Sealed Benchmark Quarantine:** `experiments/ucip14/benchmarks/sealed/` is excluded from all training scripts and dataloaders via path assertion.
2. **Cryptographic Checksum:** The sealed benchmark file hash is computed and recorded prior to any training and verified after training.
3. **Zero Overlap Guarantee:** Automated n-gram overlap check confirms 0% prompt leakage between training sets and sealed suites.
4. **Prohibition of Historical Prompt Recycling:** No prompts from historical UCIP benchmarks (`benchmark_r2_sealed.json`, `ucip11_sealed.json`, `sealed_400_raw.json`) may be reused in the UCIP-14 sealed set. All 500 sealed probes must be newly and independently authored.

---

## 4. Python Evaluation Security & Execution Containment

Evaluating model-generated Python code presents severe execution security hazards if model outputs contain malicious syscalls, infinite loops, memory bombs, or file operations.

### Critical Distinction: Subprocess Execution vs. OS Sandbox
Running code via `subprocess.run([sys.executable, tmp_path], timeout=3)` is **time-bounded subprocess execution**, NOT a full OS security containment sandbox:
- It isolates the Python interpreter process from crashing the main evaluation runner.
- It prevents infinite execution via timeouts.
- **Vulnerability:** Without OS-level containment, a subprocess inherits host file permissions and local socket capabilities.

### Mandatory Security Controls for Future Python Evaluations:
1. **Isolated Ephemeral Workspace:** Execute in a dedicated `tempfile.TemporaryDirectory` with restricted access permissions.
2. **Subprocess Resource Limits:**
   - **Timeout Ceiling:** Strict 3.0-second wall-clock execution limit (`subprocess.TimeoutExpired`).
   - **Memory Cap:** Set soft/hard address space limits (e.g., max 256 MB) to prevent Out-Of-Memory attacks.
3. **Execution Environment Stripping:**
   - Run with clean environment variables (`env={}` or minimal `PATH`/`SYSTEMROOT`).
   - Pass `-I` (isolated mode) to Python: `python.exe -I -B -s temp_script.py` to prevent loading user site-packages, environment variables, or current directory imports.
4. **AST Pre-Execution Screening:**
   - Before executing code, pass the script through Python `ast.parse`.
   - Reject scripts that import dangerous modules: `os`, `sys`, `subprocess`, `shutil`, `socket`, `urllib`, `requests`, `ctypes`, `builtins.__import__`.
   - Only allow safe builtins and standard algorithmic math operations (`math`, `typing`, `collections`).
5. **Immediate Cleanup:** Unlink temporary files in a `finally:` block regardless of whether execution succeeded, timed out, or threw a runtime error.
