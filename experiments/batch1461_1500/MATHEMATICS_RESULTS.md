# Track C: Arithmetic and Mathematical Reasoning Audit

## Batches 1461–1500: Empirical Evaluation of Quantitative Calculation

### 1. Track Objective & Setup

Track C evaluates whether the ChakrMicro neural network exhibits native mathematical computation or multi-step reasoning capabilities.

- **Dataset:** 40 mathematical tasks (`math_01` to `math_40`) divided into four balanced difficulty tiers:
  - *Tier 1: Single-Digit & Elementary Addition/Subtraction* (e.g., $147 + 258$, $1024 - 512$).
  - *Tier 2: Multiplication, Division, and Modulo* (e.g., $12 \times 12$, $50 \times 20$, $1024 / 8$, $255 \pmod{16}$).
  - *Tier 3: Applied Technical Word Problems* (e.g., node cluster workers $8 \times 4$, RAM consumption $64 - 4 - 12$, sequential API throughput).
  - *Tier 4: Multi-Step Operator Precedence & Exponentiation* (e.g., $10 + 20 \times 3$, $2^{10}$, $(50 + 50) / (2 \times 5)$).
- **Prompting Format:** Zero-shot direct question prompt:
  ```text
  Solve the following mathematical problem. Output only the final numeric answer.
  Problem: <Problem Text>
  Answer:
  ```
- **Evaluation Mechanism:** Strict numeric extraction from raw model generation with exact integer/float verification against ground truth (no substring matching).

---

### 2. Empirical Benchmark Results

| Model Evaluated | Tasks Evaluated | Correct Numerical Answers | Accuracy (%) | Format Compliance (%) | Degenerate Repetition (%) |
|---|---|---|---|---|---|
| **Canonical Baseline (Step 10,000)** | 40 | 0 | **0.00%** | 0.0% | 85.0% (34/40) |
| **Domain-Adapted Candidate (Step 10,080)** | 40 | 0 | **0.00%** | 0.0% | 77.5% (31/40) |
| **Delta ($\Delta$)** | — | 0 | **+0.00%** | 0.0% | -7.5% |

---

### 3. Failure Mode Catalog

1. **Repetitive Looping / Mode Collapse:**
   - On the Canonical Baseline, 34 tasks elicited endless loops of `The first number of the first cannot the first...` or `The first of the first...`.
   - On the Domain-Adapted Candidate, 31 tasks collapsed into repeated heading text: `- Step-by-Step Solution:\n- Step-by-Step Solution:...`.
2. **Spurious Year / Named Token Emission:**
   - On tasks with multiplication (`math_15`: $32 \times 32 = 1024$, `math_16`: $50 \times 20 = 1000$, `math_37`: $100 - 20 \times 3 = 40$), both models occasionally emitted historical years: `* January 2008 – Christian Amer` or `* John 2000 - Special number...`. Extracted numbers (`2008`, `2000`) were completely unrelated to the arithmetic.
3. **Absence of Chain-of-Thought or Step Execution:**
   - Neither model produced any intermediate arithmetic steps (e.g., carrying digits, computing intermediate products, or respecting operator precedence).
   - Neither model could compute even basic operations like $999 + 1$ or $64 \pmod 8$.

---

### 4. Scientific Conclusion

$$\text{ChakrMicro Native Mathematical Calculation Capability} = \mathbf{0.00\%} \quad [\text{NOT DEMONSTRATED}]$$

- ChakrMicro has **no learned arithmetic calculation or mathematical reasoning capability**.
- Next-token pre-training and lightweight domain adaptation do not induce numerical calculation in small transformer architectures without specialized symbolic scratchpads, tool execution gates, or fine-grained arithmetic supervision.
- **Auditor Directive:** Language modeling loss reductions (PPL $218 \to 127$) must never be conflated with mathematical capability. Any mathematical feature in ChakrView must be routed to a deterministic Python arithmetic parser.
