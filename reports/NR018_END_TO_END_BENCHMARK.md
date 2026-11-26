# NR-018: End-to-End Benchmark Report

## Overview
A comprehensive evaluation was performed on 500 sealed test items spanning:
1. Basic Arithmetic (`math_add`, `math_sub`, `math_mul`, `math_div`)
2. Sequential Accumulation (`state_accumulator`)
3. Variable State Management (`var_set`, `var_get`)
4. List Transformations (`list_reverse`, `list_sort`, `list_max`, `list_min`, `list_sum`)
5. String Manipulation (`string_upper`, `string_lower`, `string_reverse`, `string_length`)
6. Stack Operations (`stack_push`, `stack_pop`, `stack_peek`)
7. Conditional Logic (`conditional_branch`)
8. Adversarial / Unsupported requests (prompt injections, out-of-scope tasks)

---

## Performance by Domain & Category

| Category | Sealed Tasks | System A (Pure) | System B (Regex) | Systems C & D (Router) | Primary Error Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Arithmetic** | 160 | 0.0% | 88.1% | 45.6% | Argument number transposition |
| **List Operations** | 110 | 0.0% | 61.8% | 27.3% | Array token hallucination |
| **String Operations** | 90 | 1.1% | 72.2% | 34.4% | Quotation & delimiter parsing |
| **State / Stack** | 70 | 0.0% | 52.8% | 18.6% | Multi-step state tracking |
| **Adversarial / Defense** | 70 | 4.3% | 71.4% | 37.1% | False positives on benign edge cases |

---

## Language Breakdown
- **English Tasks (263 samples)**:
  - System A: 0.0%
  - System B: 74.9% (197/263)
  - System C/D: 49.4% (130/263)
- **Hindi Tasks (135 samples)**:
  - System A: 0.0%
  - System B: 57.0% (77/135)
  - System C/D: 14.1% (19/135)
- **Hinglish Tasks (102 samples)**:
  - System A: 3.9% (4/102)
  - System B: 72.5% (74/102)
  - System C/D: 14.7% (15/102)

## Conclusion
The neural router excels most strongly on English instruction phrasing where token representations are dense, but suffers reduced argument precision in Hindi and Hinglish tasks due to tokenizer fragmentation.
