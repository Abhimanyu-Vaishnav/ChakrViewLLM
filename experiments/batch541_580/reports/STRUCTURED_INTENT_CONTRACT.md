# Structured Neural Intent & Tool Contract
## Schema Specification v1.0.0 (Batches 546–550)

### 1. Architectural Intent and Scope

In ChakrView, the neural core is responsible for semantic interpretation, disambiguation, and extracting structured parameters from natural language. Symbolic computation and high-risk system operations are strictly delegated to deterministic, governed tools (`ToolRegistry` in `chakrview/runtime/tools.py`).

The **Structured Intent Contract** defines the exact protocol by which raw user queries are mapped to an unambiguous, typed machine representation.

---

### 2. Supported Intent Taxonomy

Based on the actual capabilities of the ChakrView runtime (`GroundedSessionAdapter`, `EvidenceGovernedAnsweringGate`, `ToolRegistry`):

| Intent Label | Action / Tool Target | Description & Allowed Arguments | Example User Phrasing |
| :--- | :--- | :--- | :--- |
| `calculator` | `calculator` | Evaluates deterministic arithmetic expressions.<br>**Required Args:** `expression` (str). | *"Multiply 34 by 12"*, *"What is 1500 divided by 25?"*, *"Calculate (14 + 6) * 3"* |
| `text_utility` | `text_utility` | Safe text transformations and string statistics.<br>**Required Args:** `operation` (str: `word_count`, `char_count`, `line_count`, `uppercase`, `lowercase`, `strip`), `text` (str). | *"Count words in 'quick brown fox'"*, *"Convert 'hello' to uppercase"* |
| `concept_definition`| `null` | Queries core scientific, computing, and mathematical definitions.<br>**Required Args:** `concept` (str, e.g., `cpu`, `alu`, `ram`, `pi`, `photosynthesis`). | *"Explain what an ALU does"*, *"Define RAM"* |
| `knowledge_lookup` | `retrieval` | Knowledge-base retrieval over indexed corpus chunks.<br>**Required Args:** `query` (str). | *"Find the specs for memory bus"*, *"Search docs for authentication"* |
| `direct_chat` | `null` | Social, conversational greeting, or meta dialogue.<br>**Args:** `None`. | *"Hello there"*, *"Good morning"*, *"How are you today?"* |
| `clarification_needed`| `null` | Ambiguous, underspecified, or conflicting user requests.<br>**Required Args:** `missing_field` (str), `reason` (str). | *"Calculate 45 and"* (missing operator/operand), *"Count words in"* (missing text) |
| `unsupported` | `null` | Malformed, out-of-scope, or unsafe requests (e.g. OS shell execution, malicious prompts).<br>**Required Args:** `reason` (str). | *"Delete all files in C drive"*, *"Run bash script"* |

---

### 3. Versioned Schema Format

The target generation emitted by the neural core is a single-line or compact JSON object adhering to `schema_version: "v1.0"`:

```json
{
  "schema_version": "v1.0",
  "intent": "calculator",
  "tool": "calculator",
  "args": {
    "expression": "34 * 12"
  },
  "status": "ready",
  "confidence": 1.0
}
```

#### 3.1 Field Definitions

- `schema_version`: String, must be `"v1.0"`.
- `intent`: Enum string: `["calculator", "text_utility", "concept_definition", "knowledge_lookup", "direct_chat", "clarification_needed", "unsupported"]`.
- `tool`: String or `null`. If non-null, MUST exist in `ToolRegistry` (or `"retrieval"`).
- `args`: Key-value object containing extracted arguments.
- `status`: Enum string: `["ready", "clarify", "unsupported"]`.
  - `"ready"`: All required arguments are present and valid; can proceed to tool execution.
  - `"clarify"`: Argument missing or ambiguous; runtime will ask clarification question.
  - `"unsupported"`: Out of scope or disallowed; runtime will provide safe fallback explanation.
- `confidence`: Float between `0.0` and `1.0`.

---

### 4. Security & Safety Contract

1. **Zero Raw Execution:** The runtime NEVER passes model output to `eval()`, `exec()`, `os.system()`, or shell subprocesses.
2. **Strict Tool Registry Whitelist:** `tool` can only be one of the explicitly registered tools in `ToolRegistry` (`calculator`, `text_utility`) or the retrieval engine.
3. **AST Node Safety:** The `CalculatorTool` parses expressions via strict AST whitelist (`BinOp`, `UnaryOp`, `Constant`). Any function calls, variable names, or attribute accesses are rejected before execution.
4. **DoS Prevention:** Exponentiation and magnitude limits (power $\le 100$, base $\le 10^6$) are strictly enforced.
5. **Untrusted User Input:** User text and retrieved documents are treated as untrusted data.
6. **Graceful Clarification & Fallback:** If the model predicts `"clarify"` or `"unsupported"`, the runtime abstains from execution and issues a structured, helpful explanation.
