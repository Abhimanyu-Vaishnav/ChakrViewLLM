# Architecture and Routing Contract

## Batches 1501–1540: Modular Neuro-Symbolic Tool-Use and Verification Architecture

### 1. Executive Summary & Design Principles

The Batches 1461–1500 capability audit revealed that ChakrMicro possesses genuine next-token prediction capabilities on technical text (Track A, PPL 127.70) but **0.00% native arithmetic capability** (Track C) and **0.00% parametric memory** (Track B). Factual question answering, document retrieval, citations, and unanswerable abstention are entirely driven by deterministic software components.

Batches 1501–1540 construct a modular **Neuro-Symbolic Tool-Use Layer** that:
1. Formulates an explicit, typed tool routing and verification contract.
2. Implements a safe, bounded arithmetic evaluator using AST validation (no `eval()` or unconstrained `exec()`).
3. Introduces a structured entity-attribute verification engine that explicitly distinguishes `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, and `INVALID_CLAIM`.
4. Enforces pre-emission result validation before responses reach the user.
5. Preserves strict component attribution so that deterministic tool outputs are never claimed as learned neural capabilities.

---

### 2. The Tool-Routing Interface & Decision Contract

#### 2.1 The 6 Canonical Route Types

Every incoming user request is dispatched to exactly one of six canonical routes:

| Route Type | Identifier | Operational Trigger | Delegated Component |
|---|---|---|---|
| **Direct Response** | `DIRECT_RESPONSE` | Conversational greetings, stylistic continuations, or tasks explicitly suited for pure autoregressive sequence modeling. | `ChakrMicro Neural Core` |
| **Safe Calculator** | `CALCULATOR` | Explicit or implicit quantitative calculations, numerical arithmetic (+, -, *, /, //, %, **), and word problems with unambiguous numeric formulation. | `SafeCalculatorTool` (AST Engine) |
| **Knowledge Retrieval** | `KNOWLEDGE_RETRIEVAL` | Requests for technical specifications, system parameters, architectural docs, and factual answers requiring indexed corpus lookup. | `HybridRetriever (BM25 + Dense)` |
| **Structured Verification** | `STRUCTURED_VERIFICATION` | Claims or propositions asserting that an entity has a specific attribute value, requiring validation against ground truth evidence. | `StructuredVerifier` |
| **Honest Abstention** | `ABSTAIN` | Requests lacking sufficient evidence in the indexed corpus, out-of-domain knowledge queries, or ambiguous intent. | `Threshold Gate / Policy Engine` |
| **Governed Error** | `ERROR` | Malformed syntax, arithmetic domain violations (e.g. division by zero), or tool execution exceptions. | `Error Handler` |

#### 2.2 The Observable Routing Decision Record

Every routing decision is captured in a structured `RoutingDecision` schema:

```json
{
  "request_id": "req_1501_001",
  "query": "What is 1024 / 8?",
  "route": "CALCULATOR",
  "selection_reason": "Query contains explicit arithmetic operator '/' with integer operands 1024 and 8.",
  "validated_arguments": {
    "expression": "1024 / 8",
    "normalized_expression": "1024 / 8"
  },
  "execution_status": "SUCCESS",
  "validation_status": "VALIDATED",
  "result_payload": 128,
  "provenance": {
    "component_origin": "calculator_tool",
    "citations": []
  },
  "abstention_reason": null,
  "final_response": "128"
}
```

---

### 3. Component Attribution Invariants

To prevent pseudoscience and maintain rigorous software boundaries:
1. **Arithmetic Attribution:** `calculator_tool` (Deterministic AST parsing and arithmetic execution).
2. **Factual Attribution:** `hybrid_retriever` + `deterministic_extractor` (BM25 index search and span extraction).
3. **Verification Attribution:** `structured_verifier` (Explicit entity-attribute-value matching).
4. **Abstention Attribution:** `fallback_policy` (Deterministic threshold gating).
5. **Token Generation:** `neural_core` (ChakrMicro language model).
