# NEURAL ROUTER CONTRACT & SAFETY SCHEMA
## Batches 1661–1700 Formal Specification for Neuro-Symbolic Intent & Argument Routing

### 1. Architectural Philosophy & Safety Boundaries

The neural core of ChakrView (ChakrMicro) is **strictly an intent classifier and structured argument extractor**. It possesses **zero execution privileges**:
1. The neural model **NEVER** executes calculations, Python code, shell commands, or database queries.
2. The neural model **NEVER** selects container images, network endpoints, or host file paths.
3. Every neural output must be parsed and verified against a strict **JSON Schema**.
4. Any output that fails JSON parsing, contains unknown intent keys, includes unapproved arguments, or violates type constraints is **immediately routed to `CLARIFY` or `OUT_OF_SCOPE`**.
5. All execution remains quarantined behind the existing governed subsystems: `GovernedToolRouter`, `SafeCalculator`, `StructuredVerifier`, and `SandboxedPythonTool`.

---

### 2. Supported Intent Taxonomy

The neural router must map arbitrary user utterances into exactly one of six canonical intents:

| Canonical Intent | Description | Required Argument Fields | Execution Target |
| :--- | :--- | :--- | :--- |
| **`CALCULATE`** | Mathematical evaluation request | `"expression"` (string) | `SafeCalculator.evaluate_expression` |
| **`VERIFY_CLAIM`** | Structured claim verification | `"claim"` (string), `"context"` (optional string) | `StructuredVerifier.verify_statement` |
| **`KNOWLEDGE_QUERY`**| Factual retrieval over local corpus | `"query"` (string) | Local Assistant / BM25 Index |
| **`CODE_TASK`** | Python algorithm or execution request | `"code_snippet"` (string) | `SandboxedPythonTool.execute_snippet` |
| **`CLARIFY`** | Ambiguous or incomplete user request | `"missing_field"` (string), `"reason"` (string) | User Prompt (No Tool Execution) |
| **`OUT_OF_SCOPE`** | Unsupported, hostile, or irrelevant input | `"reason"` (string) | Safe Rejection (No Tool Execution) |

---

### 3. Strict JSON Output Schema

Every candidate router generation must strictly conform to this schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ChakrViewNeuralRouterOutput",
  "type": "object",
  "properties": {
    "intent": {
      "type": "string",
      "enum": [
        "CALCULATE",
        "VERIFY_CLAIM",
        "KNOWLEDGE_QUERY",
        "CODE_TASK",
        "CLARIFY",
        "OUT_OF_SCOPE"
      ]
    },
    "arguments": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "expression": {"type": "string", "maxLength": 128},
        "claim": {"type": "string", "maxLength": 256},
        "context": {"type": "string", "maxLength": 512},
        "query": {"type": "string", "maxLength": 256},
        "code_snippet": {"type": "string", "maxLength": 1024},
        "missing_field": {"type": "string", "maxLength": 64},
        "reason": {"type": "string", "maxLength": 256}
      }
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "required": ["intent", "arguments"],
  "additionalProperties": false
}
```

---

### 4. Deterministic Validation & Fail-Closed Gate

Before any model proposal reaches the tool runtime, it undergoes the 4-stage validation pipeline:

```
Raw Model Generation String
       │
       ▼
 [ Stage 1: JSON Syntax Validation ] ───(Fail)──► Fallback to CLARIFY
       │ (Pass)
       ▼
 [ Stage 2: Intent Allowlist Check ] ───(Fail)──► Fallback to OUT_OF_SCOPE
       │ (Pass)
       ▼
 [ Stage 3: Argument Schema & Typing] ──(Fail)──► Fallback to CLARIFY
       │ (Pass)
       ▼
 [ Stage 4: Injection & Safety Sanitization ] ──(Fail)──► Fallback to OUT_OF_SCOPE
       │ (Pass)
       ▼
 Dispatch to GovernedToolRouter
```

1. **Confidence Untrustworthiness:** Model confidence scores are recorded for telemetry only; they are never used as authorization to bypass schema validation.
2. **Missing Argument Handling:** If an expression or claim is omitted (e.g. `"Please calculate this for me"`), the router must output `intent: "CLARIFY"`, `missing_field: "expression"`.
3. **Adversarial Containment:** If a user supplies instructions intended to break the parser (e.g. `"; DROP TABLE; sh -i"`), the argument extractor strips command metacharacters and quarantines execution.
