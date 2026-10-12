# ChakrView Master Sprint Report: Batches 95–98
## Operand Slot-Binding Experiment + Token-Level Checkpoint Diagnostics

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Starting HEAD**: `a0886e129f3cf3201769e4e9fe844dfc0d63f91d`  
**Ending HEAD**: (Focused local development-branch commit)  
**Host Environment**: Windows 11, Python 3.14.7, PyTorch 2.14.0+cpu, pytest 9.1.1  

---

## 1. Executive Summary & Concrete Outcomes
This sprint delivered two concrete, validated engineering deliverables:
1. **Track A (Operand Slot-Binding Experiment)**: Evaluated whether separating operand identification from tool-call syntax via structured slots (`Operand A: {a}`, `Operand B: {b}`) resolves the mode collapse observed in Batches 91–94. Evaluated on the strictly partitioned 100-pair benchmark with zero template leakage. The model reached 100% valid tool-call syntax, but shifted its constant mode attractor from `{"a": 1, "b": 1}` to `{"a": 1, "b": 8}`, maintaining a **100% constant-call collapse rate**. The result is classified honestly as **no demonstrated dynamic generalization capability**.
2. **Track B (Token-Level Checkpoint Diagnostics)**: Extended the Local Workbench with a Token-Level Diagnostic Engine. Researchers can now inspect two compatible checkpoints side-by-side, determine the exact token index of first generation divergence, compare top-K token candidate distributions and log-probabilities, evaluate teacher-forced prompt prefix log-probabilities, compute generation perplexity, and persist timestamped JSON reports.

---

## 2. Invariants & Repository State Verification
- **Canonical Model Architecture**: Strictly unchanged at 3,443,136 unique parameters (6 layers, $d_{model}=192$, $n_{heads}=6$, $hidden\_dim=512$, $vocab\_size=4096$, $max\_seq\_len=512$).
- **Canonical Step-10,000 Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified intact).
- **User Modification Preservation**: `chakrview/runtime/session.py` was left untouched and unstaged.
- **Git State**: No push to `origin`, no merge to `main`.

---

## 3. Track A: Operand Slot-Binding Results

### Experimental Setup
- **Baseline**: Canonical step-10,000 checkpoint.
- **Condition B (Reference)**: Unstructured NL tool model from Batches 91–94 (`condition_b_tool_calling_step_100.pt`).
- **Condition C (Slot-Binding SFT)**: Trained from step-10,000 for 100 optimizer steps on 71,139 tokens of structured slot-binding data (`condition_c_slot_binding_step_100.pt`, SHA-256: `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`).

### Quantitative Results Across Splits

| Metric | Condition | Train (n=64) | Val (n=18) | Test (n=18) | Clarification Control (n=5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Valid Tool Call Syntax** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | 100.0% safe non-tool |
| | Condition B (NL Tool) | 100.0% | 100.0% | 100.0% | 0.0% safe (spurious call) |
| | **Condition C (Slot-Binding)** | **100.0%** | **100.0%** | **100.0%** | 0.0% safe (spurious call) |
| **Runtime Execution Rate** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 100.0% | 100.0% | 100.0% | N/A |
| | **Condition C (Slot-Binding)** | **100.0%** | **100.0%** | **100.0%** | N/A |
| **Exact Argument Pair (A & B)**| Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 1.56% (1/64) | 0.0% (0/18) | 0.0% (0/18) | N/A |
| | **Condition C (Slot-Binding)** | **0.0% (0/64)** | **0.0% (0/18)** | **5.56% (1/18)** | N/A |
| **Constant-Call Collapse Rate**| Baseline Step 10k | 51.56% | 55.56% | 72.22% | N/A |
| | Condition B (NL Tool) | 100.0% (`{"a":1,"b":1}`) | 100.0% (`{"a":1,"b":1}`) | 100.0% (`{"a":1,"b":1}`) | N/A |
| | **Condition C (Slot-Binding)** | **100.0%** (`{"a":1,"b":8}`) | **100.0%** (`{"a":1,"b":8}`) | **100.0%** (`{"a":1,"b":8}`) | N/A |
| **End-to-End Task Success** | Baseline Step 10k | 0.0% | 0.0% | 0.0% | N/A |
| | Condition B (NL Tool) | 9.38% | 11.11% | 0.0% | N/A |
| | **Condition C (Slot-Binding)** | **10.94%** | **0.0%** | **0.0%** | N/A |

### Scientific Diagnosis
Providing explicit structured slots (`Operand A: {a}`, `Operand B: {b}`) did not resolve the constant mode collapse. Instead of dynamically extracting the operands from context, the model memorized the tool schema syntax as an invariant macro with static arguments `{"a": 1, "b": 8}`. The 1/18 test score occurred by chance because `(1, 8)` was an element of the held-out test split.

---

## 4. Track B: Workbench Token-Level Checkpoint Diagnostics

### Features
1. **Teacher-Forced Prompt Scoring**: Evaluates the mean log-probability assigned by each model to a fixed prompt prefix.
2. **First-Divergence Detection**: Identifies the exact token step at which two models diverge during free-running generation.
3. **Candidate Token Distributions**: Displays top-K candidate tokens, probabilities, and log-probabilities at the divergence point.
4. **Perplexity Scoring**: Computes the sequence perplexity on each model's continuation.
5. **Persistence & Reopening**: Saves timestamped JSON reports in `workbench/reports/` for offline inspection without re-running models.

### Verification Commands
```bash
# Inspect checkpoint metadata
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py inspect \
    --ckpt experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt

# Run token-level diagnostics
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py diagnose \
    --ckpt-a experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt \
    --ckpt-b experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt \
    --max-tokens 35

# List saved reports
python experiments/batch95_98_slot_binding_diagnostics/workbench/app.py list-reports
```

---

## 5. Focused Verification & Test Suite
- **Focused Test Suites**:
  - `tests/test_batch91_94_tool_calling_comparator.py`: 10 passed.
  - `tests/test_batch95_98_slot_binding_diagnostics.py`: 6 passed.
  - Total: **16 passed in 1.61s**.
- All tests verify slot prompt construction, zero-leakage split integrity, teacher-forced scoring, divergence detection, and report persistence.

---

## 6. Evidence-Based Next Actions
1. **Track A (Model Learning)**: Since neither freeform natural language nor explicit slot-prompting enables a 3.4M model to bind dynamic numerical arguments to a JSON schema, future efforts should investigate whether freezing attention and training a dedicated linear projection head between token embeddings and argument slots can route operands.
2. **Track B (Product)**: Add interactive token-level diff inspection directly to the Web UI canvas, allowing researchers to click on any generated token to see the full candidate probability distribution.
