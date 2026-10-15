# WP 866–871: Span-Level Argument Grounding Architecture Design

## 1. Motivation & Mathematical Formulation

The token-level pointer-generator models token transitions independently:
$$P(y_t = w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} P_{\text{ptr}}(i)$$
This formulation fails whenever an argument spans multiple subword BPE tokens (e.g. `600` $\to$ `['6', '00']`), because next-token grammar masking allows arithmetic operators before the literal completes.

### 1.1 Span Grounding Formulation
Instead of generating argument literals token-by-token from scratch or per-token pointers, the **Span-Level Grounding Architecture** decomposes plan execution into:
1. **Structural Plan Generation**: Autoregressively generate the high-level DSL structure, tools, and slot templates (e.g. `STEP 0: tool_call | calculator | args={expr=@slot0*@slot1}`).
2. **Neural Span Grounding**: For each required slot or literal entity in the prompt context $X = (x_1, \dots, x_P)$, a span extraction head predicts:
   - **Start Token Probability**: $P_{\text{start}}(i) = \text{softmax}(W_{\text{start}} h_i)$ for $i \in [0, P)$
   - **End Token Probability**: $P_{\text{end}}(j) = \text{softmax}(W_{\text{end}} [h_j; h_{\text{start}}])$ for $j \ge i$
   - **Span Type Classification**: $P_{\text{type}}(c) = \text{softmax}(W_{\text{type}} [h_i; h_j])$ where $c \in \{\text{NUMBER}, \text{STRING}, \text{IDENTIFIER}, \text{NONE}\}$
   - **Confidence Score**: $\sigma(W_{\text{conf}} [h_i; h_j])$
3. **Exact Verbatim Extraction**:
   The extracted text span is sliced directly from raw prompt source text or decoded contiguously from token span $[i, j]$, guaranteeing **zero truncation, zero rounding, and exact character preservation**.

---

## 2. Distinction & Disambiguation Contract

The architecture guarantees:
1. **Multi-token Values**: Contiguous token sequences $[i, j]$ are bound as a single atomic unit (e.g. `600` is sliced from start to end of the number span, not split).
2. **Positional Disambiguation**: Two occurrences of identical values (e.g. `14` at position 5 vs position 12) have distinct start/end index predictions.
3. **Instruction vs Argument Disambiguation**: Numbers in framing templates (e.g. "Step 1: Calculate...") have contextual representations distinguishing them from user operands.
4. **Intermediate vs Literal Distinction**: Intermediate results use `$k` syntax; span grounding is only invoked for user-grounded input arguments.
5. **Missing Argument Detection**: If maximum span confidence falls below threshold $\tau$ (e.g. 0.5) or type is `NONE`, the system flags an ambiguous/missing argument and triggers safe clarification.

---

## 3. Module Interface

The core module is implemented in [`chakrview/brain/span_grounding.py`](file:///d:/Project/ChakrView/chakrview/brain/span_grounding.py):
- `SpanGroundingHead`: Dual boundary classification + type head built on top of `ChakrMicro` hidden states.
- `ChakrSpanGroundingModel`: End-to-end model combining transformer backbone with span predictor and safe extraction methods.
- `ExtractedSpan`: Data class recording `start_idx`, `end_idx`, `span_text`, `span_type`, and `confidence`.
