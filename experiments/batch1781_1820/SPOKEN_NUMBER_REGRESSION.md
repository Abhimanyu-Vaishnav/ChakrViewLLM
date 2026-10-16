# SPOKEN NUMBER REGRESSION REPORT
## Batches 1781–1820: Expanded Isolated Spoken-Number Arithmetic Suite

### 1. Scope & Isolation Contract

As mandated by Work Package 7:
1. Spoken-number parsing is kept strictly decoupled from neural intent classification experiments and weights.
2. Spoken-number parsing is evaluated via an expanded regression suite to verify units, tens, compounds, hundreds, multiple operators, ambiguous phrasing, malformed syntax, division by zero, and injection rejection.
3. Execution correctness is strictly evaluated using `SafeCalculator.evaluate()`. Under no circumstances are `eval()` or `exec()` used.

---

### 2. Regression Results Matrix

The expanded suite (`run_spoken_number_regression.py`) tests **29 distinct cases**:
- **18 Valid Arithmetic Cases** (Units through Hundreds, Multi-Operator phrases, Compound Numbers).
- **11 Adversarial & Boundary Rejection Cases** (Decimals, negative words, non-math queries, syntax injection, division by zero, missing operands).

| Test Category | Total Test Cases | Passed Cases | Accuracy | SafeCalculator Execution Status |
| :--- | :---: | :---: | :---: | :---: |
| **Units & Single-Digit Arithmetic** | 4 | 4 | **100.0%** | All correct (`5+4=9.0`, `9-2=7.0`, `3*6=18.0`, `8/2=4.0`) |
| **Tens & Compound Numbers** | 4 | 4 | **100.0%** | All correct (`25*4=100.0`, `99-33=66.0`, `70+28=98.0`, `48/6=8.0`) |
| **Hundreds & Multi-Operator Words** | 6 | 6 | **100.0%** | All correct (`100*5=500.0`, `250+75=325.0`, `900-450=450.0`, `600/20=30.0`) |
| **Phrase Patterns ("sum of", "subtract from")** | 4 | 4 | **100.0%** | All correct (`50+25=75.0`, `12*8=96.0`, `90-40=50.0`, `100-20=80.0`) |
| **Unsupported Negatives & Decimals** | 3 | 3 | **100.0%** | Correctly rejected (Fail-closed) |
| **Division by Zero Protection** | 2 | 2 | **100.0%** | Blocked safely without exceptions |
| **Injection & Adversarial Strings** | 2 | 2 | **100.0%** | Blocked safely (`__import__` & SQL syntax rejected) |
| **Malformed Incomplete Operators** | 4 | 4 | **100.0%** | Blocked safely (Missing operands rejected) |
| **Total Test Suite** | **29** | **29** | **100.0%** | **Safe Execution: 100.0% Verified** |

---

### 3. Summary of Spoken Number Capabilities

1. **Deterministic Accuracy**: Deterministic word-to-digit conversion functions flawlessly across English numbers up to 999 and maps standard mathematical relationships cleanly to canonical expressions.
2. **Fail-Closed Guarantees**: The parser strictly rejects unsupported constructions (decimals, negative words) and passes valid canonical strings exclusively to `SafeCalculator.evaluate()`.
3. **Production Isolation**: This module remains an isolated experimental preprocessor in `experiments/batch1741_1780/` and `experiments/batch1781_1820/` and has not been integrated into the production runtime.
