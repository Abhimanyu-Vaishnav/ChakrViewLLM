# Multi-Turn Context-Reference & Memory Grounding Contract v1.0
## Batches 626–630 Specification

### 1. Architectural Purpose and Scope

In single-turn requests, all arguments reside within the user utterance (e.g. *"Calculate 17 * 23"*). In multi-turn dialogue, subsequent user turns frequently contain **anaphoric references** to entities, results, or text created in preceding turns:
- *"Now multiply that result by 4"* (refers to prior computed answer `391`)
- *"Convert the previous text to lowercase"* (refers to prior assistant-emitted text)
- *"Use the second value instead"* (refers to a specific entity among multiple candidate operands)
- *"Add 12 to it"* (ambiguous or single-candidate anaphora)

The **Context-Reference Contract** provides a typed, versioned representation for resolving antecedent references against `PersistentSessionMemory` turns.

---

### 2. Context Object Data Structure

Every turn in session history generates or contains one or more inspectable `ContextObject` records stored in turn metadata:

```json
{
  "object_id": "ctx_turn_001_ans",
  "turn_index": 1,
  "role": "assistant",
  "object_type": "number",
  "raw_content": "391",
  "origin": "calculator_tool",
  "is_tool_eligible": true,
  "created_at": 1729000000.0,
  "expires_after_turns": 5
}
```

#### 2.1 Object Type Taxonomy

| Type | Description | Example Content | Allowed Tool Targets |
| :--- | :--- | :--- | :--- |
| `number` | Computed arithmetic output or numeric argument | `"391"`, `"42"`, `"3.14159"` | `calculator` |
| `text` | Quoted user text, transformed string, or passage | `"indigenous neural core"` | `text_utility` |
| `concept` | Named scientific, architectural, or factual concept | `"alu"`, `"cpu"`, `"ram"` | `concept_definition` |
| `query` | Search query or document retrieval topic | `"memory bus width"` | `knowledge_lookup` |

---

### 3. Resolution Modes: Self-Contained vs. Anaphoric Follow-up

When a user input arrives at the neural router, the query is categorized into one of three reference states:

1. **SELF_CONTAINED (`ref_mode: "none"`):**
   - The query contains all necessary arguments within itself.
   - Example: *"Calculate 25 + 17"* -> Span extracted directly from current prompt.
2. **RESOLVED_ANAPHORA (`ref_mode: "antecedent"`):**
   - The query contains an explicit referential pronoun or phrase (`"that result"`, `"the previous answer"`, `"it"`, `"previous text"`).
   - The reference is uniquely resolved to an active, unexpired `ContextObject` in `PersistentSessionMemory`.
   - The resolved content is bound into the tool arguments.
3. **AMBIGUOUS_OR_MISSING (`ref_mode: "clarify"`):**
   - The query attempts to refer to a prior object, but:
     a) No antecedent exists (e.g. Turn 1: *"Now multiply that by 4"* with empty history).
     b) Multiple conflicting candidates exist with equal saliency without disambiguation.
     c) The referenced object has expired or been invalidated.
   - The runtime strictly **abstains from guessing** and issues a structured clarification request.

---

### 4. Security & Isolation Invariants

1. **Cross-Session Privacy:** `ContextObject` records are strictly scoped by `session_id`. Context from Session A is never accessible or resolvable in Session B.
2. **Provenance Separation:** Every resolved turn explicitly attributes the neural reference selector (`NEURAL_POINTER`), the memory retriever (`SESSION_MEMORY`), and the deterministic tool executor (`CALCULATOR_TOOL`).
3. **No Blind Defaulting:** The system never defaults to "most recent object" if the user query specifies an incompatible type (e.g. asking to multiply a text string).
