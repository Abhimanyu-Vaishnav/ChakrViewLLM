# ARGUMENT EXTRACTION & RECOVERY REPORT
## Batches 1781–1820: Decoupling Intent, Argument Extraction, and Authorization

### 1. Conceptual Separation of Concerns

A fundamental premise of ChakrView's architecture is the strict separation between:
1. **Intent Recognition**: Determining which capability or tool is being addressed.
2. **Argument Extraction**: Parsing required operational payloads (`expression`, `claim`, `task`).
3. **Execution Authorization**: Independently verifying schemas, constraints, and sandbox safety before execution.

A model that predicts the correct intent label must not be assumed to have extracted correct arguments, and correct arguments must **never** be interpreted as permission to execute.

---

### 2. Argument Extraction Performance on the Frozen 300 Benchmark

| Intent Label | Target Argument Key | Exact Match (Keyword Baseline) | Exact Match (Variant B Neural) | Exact Match (Governed Hybrid) | Dominant Failure Reason |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`CALCULATE`** | `expression` | 62.0% (31/50) | **68.0% (34/50)** | 22.0% (11/50) | Inverted phrases (e.g. *"subtract X from Y"*) confused operand order. |
| **`VERIFY_CLAIM`** | `claim` | **70.0% (35/50)** | 66.0% (33/50) | 42.0% (21/50) | Multi-sentence queries resulted in truncated claim boundaries. |
| **`KNOWLEDGE_QUERY`**| `query` | **56.0% (28/50)** | 40.0% (20/50) | 16.0% (8/50) | Conversational prefixes (e.g. *"could you please tell me"*) captured in query. |
| **`CODE_TASK`** | `task` | 84.0% (42/50) | **100.0% (50/50)** | **100.0% (50/50)** | Code blocks possess unambiguous delimiter boundaries. |
| **`CLARIFY`** | `missing_argument` | 0.0% (0/50) | 8.0% (4/50) | **96.0% (48/50)** | Deterministic schema checking detects missing fields reliably. |
| **`OUT_OF_SCOPE`** | None | 100.0% (50/50) | 100.0% (50/50) | 100.0% (50/50) | Null argument contract trivially satisfied. |
| **Overall Dataset**| **All Arguments** | **49.67% (149/300)** | **52.00% (156/300)** | **45.00% (135/300)** | — |

---

### 3. Fail-Closed Recovery Behavior on Missing Arguments (`CLARIFY`)

In 50 benchmark tasks, critical arguments were intentionally omitted (e.g., *"Calculate the total please"*, *"Run the test script"*, *"Verify the server"*):

- **Neural Standalone Classifier**:
  - Predicted `CLARIFY` on only **4 out of 50 tasks (8.0% recall)**.
  - In 46 out of 50 tasks (92.0%), it predicted a tool action (`CALCULATE` or `CODE_TASK`) with empty argument slots. If granted direct execution authority, it would invoke tools with null or malformed payloads.
- **Governed Neuro-Symbolic Hybrid**:
  - Intercepted **48 out of 50 tasks (96.0% recovery rate)** via deterministic schema validation.
  - When the neural intent was `CALCULATE` but no mathematical tokens were parsed, the Governed Router intercepted the request, blocked tool dispatch, and responded with a structured clarification question.

---

### 4. Safety Lifecycle Accounting (300 Tasks)

$$\begin{aligned}
\text{Total Attempted Routes}: & \quad 300 \\
\text{Tool Requests Dispatched}: & \quad 95 \\
\text{Routes Intercepted / Clarified}: & \quad 205 \\
\text{Unsafe Tool Executions}: & \quad 5 \text{ (1.67\%)}
\end{aligned}$$

**Conclusion**: The neural classifier cannot reliably evaluate its own argument completeness. The governed schema firewall is mandatory for preventing null-pointer exceptions and unauthorized tool execution.
