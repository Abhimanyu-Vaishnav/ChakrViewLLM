# WP 890–893: Runtime Integration Audit & Sandboxed Safety

## 1. Safety Architecture and Invariants

The span-grounding mechanism is integrated as an experimental runtime option (`SpanGroundingPlanner`) without replacing the default planner or the canonical step 10,000 baseline.

### 1.1 Invariant Enforcement
1. **Span Boundary Invariant**:
   - For any predicted span $[s, e]$, runtime asserts $0 \le s \le e < P$ where $P = \text{len}(\text{prompt\_tokens})$.
   - Out-of-bounds or reverse spans are rejected and routed to fallback clarification.
2. **Untrusted User Text**:
   - Extracted prompt text is **never** evaluated via `eval()`, `exec()`, or passed to a subshell.
   - It is parsed strictly into strongly typed literal domains (e.g. `int`, `float`, or sanitized string literals).
3. **Missing / Ambiguous Argument Guard**:
   - If model confidence $< 0.50$ or span type is `NONE`, execution is blocked and `CLARIFICATION: NEEDED` is emitted.
4. **Tool Allowlist**:
   - Tool execution remains strictly restricted to declared tool names (`calculator`, `text_utility`). Arbitrary code execution remains permanently disabled.
5. **Upstream Failure Invalidation**:
   - Downstream dependent steps (`deps=[...]`) are immediately aborted if an upstream tool dispatch fails or produces invalid output.

---

## 2. Experimental Verification

Adversarial and failure handling cases were tested to verify rejection behavior:
- **Misleading Quoted Prompts**: Successfully resolved without executing quoted text.
- **Malformed Span Offsets**: Correctly caught and sanitized.
- **Unauthorized Tool Calls**: Rejected by runtime tool validator.
- **Zero Hallucinated Execution**: All tool dispatches require explicit argument bindings.
