# Safe Arithmetic Tool Specification

## Batches 1501–1540: Restricted AST-Based Arithmetic Evaluator

### 1. Specification Overview

The `SafeCalculatorTool` provides pure, deterministic, in-memory mathematical computation for integers and finite floating-point decimals. It strictly rejects arbitrary Python code, filesystem access, process execution, external network calls, imports, and infinite/overflow DoS vectors.

---

### 2. Supported Operations & Operators

| Operation | Operator | Node Type | Example | Semantics |
|---|---|---|---|---|
| **Addition** | `+` | `ast.Add` | `147 + 258` | Exact integer or IEEE 754 float sum |
| **Subtraction** | `-` | `ast.Sub` | `1024 - 512` | Exact difference |
| **Multiplication** | `*` | `ast.Mult` | `12 * 12` | Exact product |
| **Division** | `/` | `ast.Div` | `100 / 4` | True division; returns int if exact integer result |
| **Floor Division**| `//` | `ast.FloorDiv` | `1024 // 8` | Floor integer division |
| **Modulo** | `%` | `ast.Mod` | `255 % 16` | Standard remainder |
| **Exponentiation**| `**` | `ast.Pow` | `2 ** 10` | Bounded power; base $\le 10^6$, exponent $\le 100$ |
| **Unary Plus/Minus**| `+`, `-` | `ast.UAdd`, `ast.USub`| `-42`, `-(5 + 3)` | Unary negation / identity |
| **Parentheses** | `( ... )` | Implicit in AST | `(50 + 50) / 10` | Strict mathematical precedence |

---

### 3. Security & Safety Contracts

1. **AST Node Allowlist:**
   - Only `ast.Expression`, `ast.Constant` (numeric `int`, `float`), `ast.BinOp`, and `ast.UnaryOp` are allowed.
   - Any occurrence of `ast.Call`, `ast.Name`, `ast.Attribute`, `ast.Import`, `ast.ImportFrom`, `ast.Lambda`, `ast.List`, `ast.Dict`, or statement nodes raises immediate `SecurityViolation`.
2. **Resource & DoS Limits:**
   - **Expression Length:** Maximum 256 characters.
   - **Exponent Magnitude:** Exponent absolute value $|y| \le 100$ in $x^y$.
   - **Base Magnitude:** If base $|x| > 10^6$, exponent is restricted to $\le 10$.
   - **Timeout:** Sub-millisecond evaluation limit enforced.
3. **Domain Error Handling:**
   - Division by zero (`x / 0`, `x // 0`, `x % 0`) raises `ZeroDivisionError` and returns `error_code: "DIVISION_BY_ZERO"`.
   - Overflow or non-finite values (`inf`, `nan`) return `error_code: "NON_FINITE_RESULT"`.
4. **Rounding & Precision Semantics:**
   - Does not silently round integer arithmetic.
   - Returns integers when exact (e.g. `100 / 4` $\to$ `25`, `1024 / 8` $\to$ `128`).
   - Finite decimal outputs retain up to 8 decimal places for transcendental/floating approximations without loss of significant digits.
