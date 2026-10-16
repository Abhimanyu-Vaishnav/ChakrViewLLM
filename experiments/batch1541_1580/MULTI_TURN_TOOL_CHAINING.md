# Multi-Turn Tool Chaining Specification & Integration

## Batches 1541–1580: State-Governed Arithmetic Chaining

### 1. Architectural Integration

Multi-turn tool chaining integrates `ConversationStateManager` directly into `GovernedToolRouter` and `SafeCalculator`.

```
User Turn 1: "Calculate 120 + 80."
  -> Router detects explicit arithmetic "120 + 80".
  -> SafeCalculator evaluates 120 + 80 = 200.
  -> StateManager stores ToolExecutionRecord(turn_id=1, result=200, type=NUMERIC).
  -> Output: "200".

User Turn 2: "Now divide that by 4."
  -> Router invokes StateManager.resolve_reference("now divide that by 4").
  -> Resolver finds antecedent: Turn 1 result = 200.
  -> Resolver constructs expression: "200 / 4".
  -> SafeCalculator evaluates 200 / 4 = 50.
  -> StateManager stores ToolExecutionRecord(turn_id=2, result=50, type=NUMERIC).
  -> Output: "50".
```

---

### 2. Verified Chain Operation Lifecycles

Every chained step satisfies the strict 7-phase execution lifecycle:
1. **Antecedent Resolution:** Matches referring phrases (`"that"`, `"the result"`, `"the answer"`) to the verified `ToolResultType.NUMERIC` history of the active session and tenant.
2. **Expression Validation:** Normalizes and parses the synthesized expression via `ast.parse()`.
3. **Execution Gate:** Only allowlisted operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`) are evaluated.
4. **Result Validation:** Confirms the output is a finite integer or decimal.
5. **Typed Storage:** Updates `ConversationState` with turn ID, operation metadata, and timestamps.
6. **Provenance Traceability:** Links the chained response to the antecedent turn ID.
7. **Failure Isolation:** Division by zero or syntax errors in chained operations return `CanonicalRoute.ERROR` without corrupting previous valid history.
