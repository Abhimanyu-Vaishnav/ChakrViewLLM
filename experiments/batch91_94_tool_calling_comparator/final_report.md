# ChakrView Master Sprint Report: Batches 91–94
## Validated Calculator Tool-Calling + Checkpoint Output Comparison

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Starting HEAD**: `c1e13dc1a40a8cbc7bb105bb39c05c57dac8552d`  
**Ending HEAD**: (Focused local development-branch commit)  
**Host Environment**: Windows 11, Python 3.14.7, PyTorch 2.14.0+cpu, pytest 9.1.1  

---

## 1. Executive Summary & Concrete Outcomes
This sprint delivered two concrete, validated engineering deliverables:
1. **Track A (Tool-Calling Experiment)**: Designed and evaluated an explicit, versioned, and safe deterministic calculator tool-calling protocol for `ChakrMicro v0.1`. The runtime enforces zero arbitrary execution (`eval`/`exec`), strict JSON schema validation, bounds checking, and tool-cycle caps with zero regex auto-repair. The model demonstrated **100% schema syntax acquisition** and **100% runtime execution**, but displayed **canned argument collapse** (`{"a": 1, "b": 1}`), classifying the outcome honestly as **partial capability**.
2. **Track B (Workbench Checkpoint Output Comparison)**: Extended the Local Workbench with a side-by-side Checkpoint Output Comparator. It supports inspecting two checkpoints (calculating SHA-256 and parameter counts), running side-by-side inference on identical fixed prompts, highlighting unified differences, showing deterministic oracle reference values, persisting timestamped JSON reports, and reopening reports offline without GPU/CPU recomputation.

---

## 2. Invariants & Repository State Verification
- **Canonical Model Architecture**: Strictly unchanged at 3,443,136 unique parameters (6 layers, $d_{model}=192$, $n_{heads}=6$, $hidden\_dim=512$, $vocab\_size=4096$, $max\_seq\_len=512$).
- **Canonical Step-10,000 Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified intact).
- **User Modification Preservation**: `chakrview/runtime/session.py` was left untouched and unstaged.
- **Git State**: No push to `origin`, no merge to `main`.

---

## 3. Track A: Explicit Calculator Tool Protocol & Learning Results

### Minimal Tool Protocol Schema
```json
// Model Generation (Turn 1)
TOOL_CALL: {"tool": "calculator", "op": "add", "a": 3, "b": 4}

// Runtime Deterministic Execution (Turn 2)
<|im_start|>tool
{"result": 7, "status": "SUCCESS"}
<|im_end|>

// Final Neural Generation (Turn 3)
The answer is 7.
```

### Security & Defense-in-Depth
- Zero Python `eval`/`exec`/shell execution. Pure typed integer arithmetic.
- Strict operand bounds: $[-10000, 10000]$.
- Allowlisted tools (`calculator`) and operations (`add`, `sub`, `mul`).
- Clear provenance separation: Deterministic calculator results tagged with `"provenance": "DETERMINISTIC_SAFE_CALCULATOR"`.
- Zero regex prompt-sniffing or silent argument patching.

### Evaluation Metrics on Held-Out Data

| Condition | Split | Valid Tool Syntax | Correct Tool Selection | Runtime Exec Rate | Correct Operands | End-to-End Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Step 10,000)** | Train (64) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Val (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Test (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Condition A (Direct SFT)** | Train (64) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Val (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Test (18) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Condition B (Tool-Calling SFT)**| Train (64) | **100.0%** | **100.0%** | **100.0%** | **1.56%** | **9.38%** |
| | Val (18) | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **11.11%** |
| | Test (18) | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **0.0%** |

### Scientific Finding
ChakrMicro mastered 100% compliant JSON tool invocation format from 100 SFT steps. However, operand generation collapsed to a static template `{"a": 1, "b": 1}`. The model treated the tool call as an invariant static subroutine invocation rather than binding dynamic operand slots from the context window.

---

## 4. Track B: Workbench Checkpoint Output Comparator

### Deliverables
- **CLI & Web Engine**: [`experiments/batch91_94_tool_calling_comparator/workbench/app.py`](file:///d:/Project/ChakrView/experiments/batch91_94_tool_calling_comparator/workbench/app.py).
- **Features**:
  1. Inspects checkpoints for SHA-256, parameter count (deduplicating tied embeddings), step count, and metadata.
  2. Runs side-by-side deterministic inference with identical tokenizers and decoding limits.
  3. Displays unified diffs and tags outputs as `IDENTICAL` or `DIVERGENT`.
  4. Displays deterministic oracle references for arithmetic prompts.
  5. Saves timestamped JSON comparison reports and reloads them offline without re-running models.
  6. Graceful error handling for missing or corrupted checkpoints without crashing.

### Verification Commands
```bash
# Inspect checkpoint
python experiments/batch91_94_tool_calling_comparator/workbench/app.py inspect --ckpt experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt

# Run side-by-side comparison
python experiments/batch91_94_tool_calling_comparator/workbench/app.py compare --ckpt-a experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt --ckpt-b experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt --max-tokens 40

# List persisted reports
python experiments/batch91_94_tool_calling_comparator/workbench/app.py list-reports
```

---

## 5. Verification & Test Suite
- **Focused Test Suite**: [`tests/test_batch91_94_tool_calling_comparator.py`](file:///d:/Project/ChakrView/tests/test_batch91_94_tool_calling_comparator.py) — 10 passed in 2.27s.
  - Safe calculator operations, bounds checks, error codes.
  - Checkpoint inspection, parameter count verification.
  - Deterministic oracle logic.
  - Report persistence and offline reopening.
- **Historical Regression Suite**: 100% passing across existing regressions.

---

## 6. Evidence-Based Next Actions
1. **Track A (Tool-Calling)**: To overcome canned argument collapse without scaling parameter count, explore explicit prefix-slot prompting (e.g. `User: Calculate 3 + 4. Operands: [3, 4] -> Call: {"a": 3, "b": 4}`) to simplify cross-attention operand extraction.
2. **Track B (Workbench)**: Add visual attention weight heatmaps or token generation probability distributions to the Checkpoint Comparator view to help diagnose divergence at specific token positions.
