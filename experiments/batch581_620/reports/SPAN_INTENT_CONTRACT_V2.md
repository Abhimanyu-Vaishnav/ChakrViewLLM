# Neural Span Extraction & Intent Contract v2.0
## Schema Specification (Batches 586–590)

### 1. Conceptual Framework

In Batches 541–580, we established that a 3.4M parameter Transformer core reliably classifies semantic intent and emits structured syntax, but suffers from severe token-drifting when attempting to regenerate user-provided operands and strings character-by-character (3.4% argument accuracy).

The **Span Extraction & Intent Contract v2.0** solves this by formalizing argument extraction as **Source-Span Indexing**:
Instead of generating the string `"94 - 62"`, the model identifies the precise contiguous span in the user's input sequence:
- `span_start`: index of the first token/character of the argument
- `span_end`: index of the last token/character of the argument (exclusive)
- `extracted_value`: the exact substring sliced from `user_query[span_start:span_end]`

---

### 2. Supported Task Contract & Field Specification

```json
{
  "schema_version": "v2.0",
  "intent": "calculator",
  "tool": "calculator",
  "spans": {
    "expression": {
      "start_char": 21,
      "end_char": 28,
      "text": "94 - 62"
    }
  },
  "status": "ready"
}
```

#### 2.1 Intent & Span Fields

| Intent | Tool | Required Span Field | Source Substring Example |
| :--- | :--- | :--- | :--- |
| `calculator` | `calculator` | `expression` | `"94 - 62"`, `"52 + 67"`, `"(55 + 37) - 14"` |
| `text_utility` | `text_utility` | `operation`, `text` | `text`: `"'indigenous neural core'"`, `operation`: normalized (`"uppercase"`, `"word_count"`) |
| `concept_definition`| `null` | `concept` | `"cpu"`, `"alu"`, `"ram"`, `"fpga"`, `"tpu"` |
| `knowledge_lookup` | `retrieval` | `query` | `"memory bus width specifications"`, `"user authentication policy"` |
| `direct_chat` | `null` | `None` | (No spans needed) |
| `clarification_needed`| `null` | `missing_field` | (Identifies which required span is missing) |
| `unsupported` | `null` | `None` | (Flagged as disallowed) |

---

### 3. Span Verification & Safety Protocol

1. **Exact Substring Verification:**
   Before any tool execution, the runtime verifies:
   $$\text{extracted\_value} == \text{user\_query}[\text{start\_char} : \text{end\_char}]$$
   If this assertion fails, the span proposal is rejected as out-of-bounds.
2. **Deterministic Tool Guardrails:**
   For `calculator`, the extracted substring is parsed through `CalculatorTool` AST validation. Malicious commands or non-mathematical AST nodes are strictly rejected.
3. **No Hallucinated Operands:**
   Because arguments are anchored directly to user query spans, the model cannot invent or substitute training operands (e.g. `22 * 15`) in place of unseen user numbers (`94 - 62`).
