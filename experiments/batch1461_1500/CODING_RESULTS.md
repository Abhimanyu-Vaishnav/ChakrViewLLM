# Track D: Coding and Program Execution Audit

## Batches 1461–1500: Empirical Evaluation of Python Function Synthesis

### 1. Track Objective & Setup

Track D tests whether ChakrMicro can synthesize syntactically valid and functionally correct Python code when prompted with function signatures and docstrings.

- **Dataset:** 40 isolated Python coding tasks (`code_01` to `code_40`) covering:
  - *Elementary logic:* `add(a, b)`, `is_even(n)`, `square(x)`, `double_number(n)`, `is_positive(x)`.
  - *String operations:* `string_length(s)`, `greet(name)`, `reverse_string(s)`, `count_vowels(s)`.
  - *List & data manipulation:* `get_first_element(lst)`, `sum_list(numbers)`, `find_max(numbers)`, `filter_positive(numbers)`.
  - *Data structure lookups & algorithmic logic:* `get_dict_value(d, key, default)`, `is_palindrome(s)`, `factorial(n)`, `binary_search(arr, target)`.
- **Execution Sandbox:** Restricted sub-process execution environment with strict 2.0-second timeouts, resource constraints, blocked networking, and isolated namespaces.
- **Evaluation Criteria:**
  1. *Syntax Validity Rate:* Generated code must parse into a valid Python AST without `SyntaxError` or `IndentationError`.
  2. *Unit Test Pass Rate:* Generated function must pass 3–5 unit test assertions without throwing runtime exceptions.
  3. *Unassisted Completion:* Evaluates raw model output; no manual repairs or automated code formatters.

---

### 2. Empirical Benchmark Results

| Model Evaluated | Tasks Evaluated | Valid Syntax Count | Syntax Validity Rate (%) | Unit Tests Passed | Test Pass Rate (%) | Primary Failure Mode |
|---|---|---|---|---|---|---|
| **Canonical Baseline (Step 10,000)** | 40 | 5 | **12.50%** | 0 | **0.00%** | Unterminated docstrings / no indented block |
| **Domain-Adapted Candidate (Step 10,080)** | 40 | 2 | **5.00%** | 0 | **0.00%** | Unterminated docstrings / syntax errors |
| **Delta ($\Delta$)** | — | -3 | **-7.50%** | 0 | **+0.00%** | N/A |

---

### 3. Detailed Failure Analysis

#### A. The "Unterminated Triple-Quote" Trap
In over 65% of coding prompts, the model began by attempting to write a docstring, but immediately hallucinated corrupted conditionals inside quotation marks:
```python
def add(a, b):
    """
    if not isinstance(")
    if not self._subset(")
    if not self._subset(")
```
Because the string literal was never closed (`SyntaxError: unterminated triple-quoted string literal`), the resulting code failed AST compilation.

#### B. Indentation & Pass Collapses
In cases where valid syntax was achieved (5/40 on Base, 2/40 on Candidate):
```python
def get_first_element(lst):
    pass
    pass
    pass
    # Check the result of the result...
```
While syntactically valid (due to `pass`), it returns `None`, immediately failing unit tests (`assert get_first_element([1, 2, 3]) == 1` -> `AssertionError`).

#### C. Inability to Generate Algorithmic Logic
Neither model produced working logic for basic arithmetic operations (`return a + b`), string operations (`return len(s)`), or boolean conditions (`return n % 2 == 0`).

---

### 4. Scientific Conclusion

$$\text{ChakrMicro Native Code Synthesis Capability} = \mathbf{0.00\%} \quad [\text{NOT DEMONSTRATED}]$$

- ChakrMicro possesses **zero functional Python code generation capability**.
- Pre-training on mixed technical text and markdown does not confer code execution or function completion capability at this parameter scale.
- **Auditor Directive:** ChakrView must never rely on ChakrMicro to generate executable code, scripts, or runtime queries. Any code execution feature must rely strictly on deterministic templates and external sandboxes.
