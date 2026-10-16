# Dataset Split & Leakage Audit

## 1. Audit Methodology
To guarantee evaluation integrity, an automated audit script was executed against `train.jsonl`, `val.jsonl`, and `test_held_out.json`.
The audit verified:
1. Exact string matches between training and held-out test prompts.
2. Direct answer leakage across splits.
3. Near-duplicate template repetition.
4. Overlap in numerical targets.

---

## 2. Leakage Verification Results

| Audit Check | Train vs Val | Train vs Test Held-Out | Verdict |
|:---|:---:|:---:|:---:|
| **Exact Prompt Overlap** | 0 / 10 | 0 / 150 | **ZERO LEAKAGE (PASS)** |
| **Exact Sentence Overlap** | 0 / 10 | 0 / 150 | **ZERO LEAKAGE (PASS)** |
| **Arithmetic Operands (25–54)** | None in Train (max 20) | Range 25–54 | **OUT-OF-DISTRIBUTION (PASS)** |
| **Compositional Target ($X \rightarrow Z$)** | Hop 1 & Hop 2 separate | Direct composite chain | **COMPOSITIONAL TEST (PASS)** |
| **Counterfactual Services** | None in Train | 15 unmentioned services | **COUNTER-CONTROL (PASS)** |

---

## 3. Findings
- Zero training examples leaked into the held-out test suite.
- The arithmetic test specifically probes out-of-distribution numeric induction (25 to 54) rather than trained integers.
- The compositional test strictly probes transitive inference ($A \rightarrow B$ and $B \rightarrow C$ taught separately; $A \rightarrow C$ evaluated).
