# NR-020: Source-Grounding Implementation Forensics

## 1. Architectural Component Dissection
An audit of `chakrview/runtime/source_grounding.py` was performed to identify the exact origin of parameter extraction:

| Operation Class | Operation Name | Origin of Operation Key | Origin of Parameters / Arguments |
| :--- | :--- | :--- | :--- |
| **Arithmetic** | `math_add`, `math_sub`, `math_mul`, `math_div` | **Neural Router** | **Deterministic regex / normalizer** (`extract_numbers_with_spans`) |
| **List Operations** | `list_reverse`, `list_sort`, `list_max`, `list_min`, `list_sum` | **Neural Router** | **Deterministic bracket regex** (`extract_array_with_span`) |
| **String Operations**| `string_upper`, `string_lower`, `string_reverse`, `string_length`| **Neural Router** | **Deterministic quote regex** (`re.search(r"['\"]([^'\"]+)['\"]")`) |
| **State / Stack** | `stack_push`, `var_set` | **Neural Router** | **Deterministic regex** |
| **Conditionals** | `conditional_branch` | **Neural Router** | **Deterministic regex & threshold compare** |

---

## 2. Core Forensic Insight: Division of Responsibility
- **The Neural Router is responsible for:**
  - Semantic classification of user intent into one of 21 operation strings (`"operation": "math_mul"`).
  - Multilingual natural language phrasing comprehension (mapping *"को ... से गुणा करो"* or *"into ... kitna hoga"* to `math_mul`).
  - Adversarial intent classification (mapping prompt injections and off-topic questions to `status: "unsupported"`).
- **The Deterministic Grounder is responsible for:**
  - Finding numeric literals, Devanagari numerals, or quoted substrings.
  - Converting substrings into exact typed Python objects (`int`, `float`, `list`, `str`).
  - Binding parameters to the target schema keys (`a`, `b`, `list`, `string`).

**Conclusion**: The neural model **does not extract individual character span indices**. It predicts the *operation name*, while deterministic parsing code extracts the arguments from the source text. Therefore, the 50.5% benchmark performance is a true **neuro-symbolic collaboration**: neural understanding supplies the semantic operation, while deterministic code supplies exact numerical grounding.
