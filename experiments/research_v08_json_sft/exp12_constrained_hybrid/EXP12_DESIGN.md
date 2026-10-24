# ChakrView Experiment 12 — Schema-Constrained Hybrid Extraction Design Specification

## 1. Mission and Research Objective

ChakrView Experiment 11 revealed that:
1. Combining the native causal LM head with the span-pointer achieved high JSON parse validity (85–100%) and successfully copied multi-token strings into slots.
2. However, the pretrained causal LM head lacked autoregressive state discipline: once a field closed, it repeatedly looped on identical keys (e.g., `{"status": "...", "status": "..."}`), leading to severe schema invalidity and whole-object exact match collapse (5% Val, 1.7% Test A, 0% Test B).

**Experiment 12 Objective**:
Determine whether a **schema-constrained decoding controller** paired with the frozen Experiment 11 checkpoint and pointer parameters resolves this structural failure mode, isolates true value selection accuracy, and achieves reliable structured JSON extraction across familiar and held-out entities.

---

## 2. Phase 0 Audit Summary: Experiment 11 Architecture & Provenance

### 2.1 Model Architecture & Parameter Accounting
- **Backbone**: `ChakrMicro` 6-layer Transformer ($d_{\text{model}}=192$, $n_{\text{heads}}=6$, tied embedding LM head $V=4096$). Total backbone parameters: ~3.15M.
- **Tied LM Head**: Shares weights directly with `embedding.weight` (`[4096, 192]`).
- **Span Pointer Projections**:
  - `query_start.weight`: `[192, 192]` (36,864 parameters)
  - `key_start.weight`: `[192, 192]` (36,864 parameters)
  - `query_end.weight`: `[192, 192]` (36,864 parameters)
  - `key_end.weight`: `[192, 192]` (36,864 parameters)
  - Total pointer parameters: 147,456.

### 2.2 Parameter Provenance & Freeze State
- Loaded from Canonical `Step-10000`: Transformer backbone embedding and layers 0–5.
- Loaded from `exp10_span_pointer_best.pt`: `query_start`, `key_start`, `query_end`, `key_end`.
- Trained in Exp 11: 50 pilot overfit steps on an 8-record batch to verify loss and backprop stability.
- Frozen in Exp 12: **No parameters retrained**. Evaluated on the exact frozen checkpoint `exp11_hybrid_best.pt`.

### 2.3 Task Instruction & Required Fields Mapping
The prompt contains explicit instructions specifying the target fields:
- `"Extract entity, status, and code as JSON."` $\rightarrow$ `["entity", "status", "code"]`
- `"Extract entity and status as JSON."` $\rightarrow$ `["entity", "status"]`
- `"Extract status and code as JSON."` $\rightarrow$ `["status", "code"]`
- `"Extract entity and code as JSON."` $\rightarrow$ `["entity", "code"]`
- `"Extract status as JSON."` $\rightarrow$ `["status"]`
- `"Extract entity as JSON."` $\rightarrow$ `["entity"]`

---

## 3. Schema Controller Design (Finite-State Machine)

The Schema Controller enforces a deterministic finite-state transition grammar for structured JSON synthesis:

### 3.1 Formal States
1. `STATE_START`: Emits `{"`.
2. `STATE_KEY`: Emits `"key": "` for the next requested field in canonical order (`entity`, `status`, `code`).
3. `STATE_VALUE`: Invokes the Span Pointer on the prompt context under pure inference masking to extract the contiguous slot span.
4. `STATE_CLOSE_VALUE`: Emits closing quote `"`.
5. `STATE_DELIMITER`: If further requested fields remain, emits `", "`; else emits `}` and transitions to `STATE_DONE`.
6. `STATE_DONE`: Terminates generation cleanly.

### 3.2 Formal Guarantees
- **Never emits duplicate keys**: Tracked via `remaining_fields` queue.
- **Never emits unrequested keys**: Only fields extracted from the instruction are enqueued.
- **Never omits requested keys**: Every enqueued field must be visited before closing `}`.
- **Enforces valid JSON punctuation & escaping**: Quotes, colons, commas, and braces are deterministically placed.
- **Pure Model Value Attribution**: The controller never injects ground-truth values. The value is strictly selected by the pointer model (or oracle in the upper-bound arm).

---

## 4. Experimental Arms

Evaluated across Validation (60), Test A (60), and Test B (60):
- **Arm A (Native LM Baseline)**: Unconstrained autoregressive generation from canonical `Step-10000`.
- **Arm B (Unconstrained Exp 11 Hybrid)**: Frozen `exp11_hybrid_best.pt` with unconstrained LM token routing.
- **Arm C (Constrained Hybrid)**: Frozen `exp11_hybrid_best.pt` with Schema Controller active.
- **Arm D (Oracle-Value Upper Bound)**: Schema Controller with ground-truth source spans directly supplied.
- **Arm E (Deterministic Regex Baseline)**: Rule-based regex extraction sanity check.
