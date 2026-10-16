# Conversation State Specification

## Batches 1541–1580: Multi-Turn Tool Result State and Anaphora Resolution

### 1. Specification Overview

The `ConversationStateManager` maintains an isolated, typed memory of prior tool execution outcomes for multi-turn sessions. It resolves anaphoric references (e.g. `"now divide that by 4"`, `"add 50 to the result"`) directly to validated antecedent tool outputs rather than attempting to parse guessed values from conversational text.

---

### 2. State Invariants & Isolation Contract

1. **Strict Tenant & Session Keying:**
   - Every session is uniquely keyed by `(tenant_id, conversation_id)`.
   - Tool execution history is strictly partitioned; state cannot leak across tenant boundaries or between separate conversations of the same tenant.
2. **Typed Tool Execution Records:**
   - Results are stored as structured `ToolExecutionRecord` instances with explicit `ToolResultType` (`NUMERIC`, `TEXT`, `VERIFICATION_RECORD`, `ERROR`).
   - Only records with `success == True` and matching expected result types are eligible for anaphora substitution.
3. **Bounded Memory & Chain Limits:**
   - Configurable `max_chain_length` (default 10 turns) enforced via FIFO sliding window to prevent unbounded memory growth.
   - Time-to-Live (`ttl_seconds`, default 3600s): Expired sessions are automatically reset on access.
4. **Conversation Resets:**
   - Calling `reset_session()` clears all stored tool outcomes. Following a reset, queries with `"that"` or `"the previous result"` safely fail resolution and abstain rather than reusing stale state.
5. **Ambiguity Handling:**
   - If a query refers ambiguously to multiple possible earlier values, the resolver surfaces `ambiguous = True` or defaults to the strictly most recent antecedent with documented provenance.
