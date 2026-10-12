# Batch 101: Interactive Token Diagnostics in Local Workbench

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Deliverable Path**: `experiments/batch99_102_representation_audit/workbench/app.py`  
**Reports Storage**: `experiments/batch99_102_representation_audit/workbench/reports/`  

---

## 1. Overview & Architectural Design

Batch 101 upgrades the Local Workbench from static text side-by-side comparison to an **interactive, token-level inspection system** without introducing any external web frameworks or heavy dependencies:
- Built strictly with Python standard library `http.server`, PyTorch, and vanilla HTML5/CSS/JavaScript.
- Reuses the project's indigenous BPE tokenizer artifacts (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
- Preserves 100% backward compatibility with CLI workflows and saved JSON reports from Batches 91–98.

---

## 2. Interactive Features Implemented

### 2.1 Token Boundary Chip Rendering
- Generated outputs for both Reference (Model A) and Candidate (Model B) are rendered as individual, hoverable token chips.
- Whitespace, special tokens (`<|im_start|>`, `<|im_end|>`, `\n`), and byte subwords are decoded cleanly without raw unescaped HTML injections or UI breakage.

### 2.2 Interactive Token Inspector Drawer
Clicking on any generated token chip displays an instant floating diagnostic panel:
- **Token Identity**: Integer Token ID and string representation.
- **Sequence Indices**: Generation step index and absolute sequence position.
- **Origin Model**: Explicit label showing which checkpoint generated the token.
- **Top-5 Candidate Distribution**: Rank, token string, token ID, assigned probability ($P$), and log-probability ($\log P$) recorded during inference.
- **Divergence Landmark**: Clear visual indicator (`⚠️ FIRST DIVERGENCE POINT`) marking the exact sequence step where two models branched.

### 2.3 Transparent Methodological Labeling
The interface explicitly displays two methodological distinctions:
1. **Teacher-Forced vs Free-Running**:
   - Fixed prompt prefix scores are labeled as `TEACHER-FORCED PROMPT LOG-PROB` evaluated on identical input prefixes.
   - Continuations are labeled as `FREE-RUNNING GENERATION` generated autoregressively by each model.
2. **Uncalibrated Model Scores Disclaimer**:
   - The UI displays an explicit persistent banner reminding researchers:
     > *"Probabilities and log-probabilities are raw model score densities, NOT calibrated confidence or guarantees of factual truth."*

### 2.4 Offline Persistence & Legacy Compatibility
- All top-5 candidate distributions are persisted into the timestamped diagnostic report JSON.
- Saved reports can be reopened offline via `Reopen Saved Report (Offline)` or CLI without requiring models or GPUs in memory.
- If an older report without top-5 candidates is opened, the UI cleanly displays `Candidate details unavailable in this report version` rather than fabricating values or throwing exceptions.

---

## 3. CLI Commands Verified

```bash
# 1. Inspect checkpoint metadata & SHA-256
python experiments/batch99_102_representation_audit/workbench/app.py inspect \
    --ckpt experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt

# 2. Run token-level diagnostic comparison
python experiments/batch99_102_representation_audit/workbench/app.py diagnose \
    --ckpt-a experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt \
    --ckpt-b experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt \
    --max-tokens 35

# 3. List saved reports
python experiments/batch99_102_representation_audit/workbench/app.py list-reports

# 4. Launch interactive browser UI
python experiments/batch99_102_representation_audit/workbench/app.py serve --port 8501
```
