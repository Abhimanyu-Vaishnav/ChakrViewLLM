# Runtime Integration Audit: Batches 741–780
## Autoregressive Plan DSL, Variable-Length Neural Planning & Safe Execution

**Audit Date:** 2026-10-15  
**Candidate:** `candidate_step250_autoregressive_planner.pt`  

---

### 1. Multi-Stage Pipeline Separation

In compliance with non-negotiable principle #8, the runtime separates plan generation into auditable, isolated stages:

1. **Neural Generation:** Autoregressively generates raw token sequences until `<END_PLAN>` or maximum token budget (150 tokens).
2. **DSL Parsing (`PlanDSLParser`):** Parses raw strings into typed structures (`ParsedStep`, `ParsedPlan`) and normalizes variable bindings (`$0`).
3. **Graph & DAG Validation (`PlanDSLValidator`):** Enforces DAG acyclicity ($dep < step\_id$), verifies presence of terminal `<END_PLAN>`, and rejects illegal actions.
4. **Tool Authorization:** Rejects unauthorized tools (e.g. `root_exploit`, `shell_cmd`) before tool execution.
5. **Execution (`ToolExecutor`):** Safely evaluates AST arithmetic and text manipulation.
6. **Provenance Reporting:** Logs telemetry distinguishing neural predictions from deterministic AST execution and validator gating.

---

### 2. Safeguards Against Truncated or Hallucinated Plans

- **Terminal Marker Invariant:** Any plan missing `<END_PLAN>` is flagged as truncated and rejected by `PlanDSLValidator.validate()`.
- **Zero Gold Substitution:** If the model emits a malformed DSL line, the validator returns `is_valid: False` and prevents tool dispatch. The runtime **never** substitutes a pre-scripted gold plan.
