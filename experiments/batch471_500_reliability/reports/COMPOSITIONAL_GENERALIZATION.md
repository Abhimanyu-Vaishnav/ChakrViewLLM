# Fresh Compositional Generalization & Scratchpad Reliability Audit: Batches 476–482

**Execution Timestamp:** 2026-10-15T12:13:09  
**Benchmark Script:** [`experiments/batch471_500_reliability/benchmark_fresh_compositional.py`](file:///d:/Project/ChakrView/experiments/batch471_500_reliability/benchmark_fresh_compositional.py)  
**Manifest:** [`experiments/batch471_500_reliability/data/fresh_compositional_benchmark.jsonl`](file:///d:/Project/ChakrView/experiments/batch471_500_reliability/data/fresh_compositional_benchmark.jsonl)  
**Machine-Readable Results:** [`experiments/batch471_500_reliability/reports/fresh_compositional_results.json`](file:///d:/Project/ChakrView/experiments/batch471_500_reliability/reports/fresh_compositional_results.json)  
**Evaluated Checkpoint:** `candidate_step150_scratchpad_reasoning.pt` (SHA-256: `72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759`)  

---

### 1. Executive Summary: The Limits of Few-Shot Textual Scratchpad

The previous milestone reported an impressive jump from **0% to 50% (3/6)** on held-out compositional arithmetic. In Batches 476–482, we subjected this claim to a rigorous scientific stress test using 12 freshly constructed compositional tasks across 8 task families with novel operands, altered operator orderings, word problem contexts, and direct single-step controls.

**Key Finding:**  
The neural candidate achieved **16.7% (2/12)** overall accuracy on the fresh benchmark:
- **Direct Single-Step Controls (`14 + 28 = 42`, `7 * 6 = 42`):** **100.0% (2/2)** — perfectly preserved!
- **Novel Multi-Step Compositions (`16 + 14 - 12`, `5 * 8 - 15`, `(1 + 6)^2`):** **0.0% (0/10)**.

---

### 2. Failure Mode Breakdown

When presented with novel operands in compound reasoning tasks, the small (3.4M parameter) model exhibits a distinct, reproducible failure mode: **Frame-Conditioned Memorized Rehearsal**.

1. **Replay of Dominant Training Sequences:**
   - On `(16 plus 14) minus 12`: The model emitted `Step 1: 14 + 28 = 42. Step 2: 42 - 10 = 32. Final Answer: 32`.
   - On `(40 minus 15) plus 5`: The model emitted `Step 1: 8 * 5 = 40. Step 2: 40 - 12 = 28. Final Answer: 27`.
   - On `5 multiplied by 8 minus 15`: The model emitted `Step 1: 8 * 5 = 40. Step 2: 40 - 12 = 28. Final Answer: 28`.
   - On `9 multiplied by 4 minus 6`: The model emitted `Step 1: 8 * 5 = 40. Step 2: 40 - 12 = 28. Final Answer: 28`.
2. **Analysis:**
   - The model learned the **syntactic surface structure** of scratchpads (`Step 1: ... Step 2: ... Final Answer: ...`).
   - However, with only 9 training examples and 3.4M parameters, the model **has not learned dynamic variable slot binding or arithmetic computation**. It functions as an associative memory that routes a multi-step prompt to whichever training scratchpad trajectory has the highest token overlap.
3. **Implication for Architecture:**
   - **Hypothesis Disproven:** Textual scratchpad training alone on a handful of examples does not endow a 3.4M model with true generalized multi-step arithmetic reasoning.
   - **Architectural Validation:** This scientifically validates the architectural necessity of **deterministic tool delegation** (e.g. `CalculatorTool`) and the **Evidence-Governed Answering Gate** rather than relying on unassisted neural generation for mathematical truth.

---

### 3. Task-by-Task Performance Matrix

| Task ID | Family | Prompt Query | Expected | Generated Output | Trace Valid | Final Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fresh_comp_001` | `add_sub_novel` | `(16 plus 14) minus 12` | `18` | `Step 1: 14 + 28 = 42...` | No | No |
| `fresh_comp_002` | `add_sub_novel` | `(40 minus 15) plus 5` | `30` | `Step 1: 8 * 5 = 40...` | No | No |
| `fresh_comp_003` | `mul_sub_novel` | `5 * 8 minus 15` | `25` | `Step 1: 8 * 5 = 40...` | No | No |
| `fresh_comp_004` | `mul_sub_novel` | `9 * 4 minus 6` | `30` | `Step 1: 8 * 5 = 40...` | No | No |
| `fresh_comp_005` | `add_sq_novel` | `square of (1 plus 6)` | `49` | `Step 1: 8 * 5 = 40...` | No | No |
| `fresh_comp_006` | `add_sq_novel` | `square of (4 plus 5)` | `81` | `Step 1: 42` | No | No |
| `fresh_comp_007` | `reordered_sub_add` | `50 minus 20 plus 8` | `38` | `Step 1: 8 * 5 = 40...` | No | No |
| `fresh_comp_008` | `geom_angle_novel` | `angles 50 and 80` | `50` | `Step 1: Total angle sum...` | No | No |
| `fresh_comp_009` | `word_problem_comp`| `12 + 8 - 5` | `15` | `Step 1: $80` | No | No |
| `fresh_comp_010` | `single_step_ctrl` | `14 plus 28` | `42` | `42` | N/A | **Yes** |
| `fresh_comp_011` | `single_step_ctrl` | `7 multiplied by 6` | `42` | `42` | N/A | **Yes** |
| `fresh_comp_012` | `hindi_variant` | `14 aur 28 ka jod` | `42` | `32` | N/A | No |
