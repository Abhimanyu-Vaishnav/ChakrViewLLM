# FAILURE ANALYSIS & TAXONOMY
## Batches 1661–1700 Root Cause Diagnosis of Neural Routing Failures

### 1. Classification of Model Failures on Held-Out Tasks

Out of 120 held-out tasks evaluated against the Candidate Neural Router, 97 tasks (80.83%) failed to achieve correct intent and argument parsing.

```
Total Held-Out Tasks: 120
  ├── Passed (Intent & Argument Correct): 23 (19.17%)
  └── Failed Tasks: 97 (80.83%)
        ├── FM-1: Non-JSON / Syntax Truncation (85.8% of raw outputs)
        ├── FM-2: Inability to Transform Natural Operators to Math Symbols
        ├── FM-3: Entangled Entity Extraction
        └── FM-4: Premature EOS Generation
```

---

### 2. Failure Mode Details and Raw Model Outputs

#### FM-1: Non-JSON / Syntax Truncation (85.83% of outputs)
- **Description:** Model begins generating JSON but fails to close quotes or curly braces, or reverts back to natural prose.
- **Task ID:** `test_calc_001`
- **Prompt:** `"User: I need to find the sum of 512 and 1024\nRouter: "`
- **Raw Candidate Generation:**
  `"User: I need to find the sum of 512 and 1024\nRouter: {"intent": "OUT_OF_SCOPE", "arguments": {"reason": "No intent..."` (truncated without closing `}`)
- **Root Cause:** Without a constrained grammar decoder (e.g., Trie-based or EBNF-guided token masking), a 3.4M causal LM cannot reliably maintain the bracket balance required for valid JSON.

#### FM-2: Inability to Transform Natural Language Operators to Math Symbols
- **Description:** Model cannot convert phrase `"Multiply 23 by 17"` into `"23 * 17"`.
- **Task ID:** `test_calc_002`
- **Prompt:** `"Multiply 23 by 17 please"`
- **Expected:** `{"intent": "CALCULATE", "arguments": {"expression": "23 * 17"}}`
- **Candidate Behavior:** Emits text drift or copies the literal prompt without extracting numeric values.
- **Root Cause:** Symbol manipulation requires pointer mechanisms or arithmetic AST builders that pure autoregressive next-token prediction does not possess.

#### FM-3: Entangled Entity Extraction
- **Description:** In `VERIFY_CLAIM` tasks, the model frequently conflates the subject entity with the claim predicate.
- **Task ID:** `test_verify_004`
- **Prompt:** `"I heard that database connections terminate after 60s. Is this true?"`
- **Expected:** `{"intent": "VERIFY_CLAIM", "arguments": {"claim": "database connections terminate after 60s"}}`
- **Candidate Behavior:** Generates empty arguments or places conversational commentary inside the claim argument.

---

### 3. Engineering Recommendations

| Observed Failure | Ineffective Remedy | Recommended Effective Architecture |
| :--- | :--- | :--- |
| **JSON Syntax Errors** | Increasing learning rate or training epochs | **Grammar-Constrained Decoding / JSON Masking** |
| **Argument Extraction Failure** | More synthetic text | **Pointer-Generator Networks / Span Extraction** |
| **Latency Penalty (259 ms vs 0.06 ms)** | CPU multithreading | **Fast Heuristic / Classifier Head (Linear Readout)** |

A classification head (e.g. `[CLS]` token $\rightarrow$ 6-way linear layer) would execute in $<1$ ms on CPU and guarantee 100% valid intent selection, completely bypassing the brittleness of autoregressive JSON text synthesis.
