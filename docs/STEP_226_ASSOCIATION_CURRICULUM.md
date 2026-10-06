# Step 226: Association Curriculum Ladder

## 1. Scientific Objective
Evaluate progression across a sequential 9-level gated curriculum:
- **L0**: One association (`B -> Y`, query `B`)
- **L1**: Two associations (`A -> X, B -> Y`, query `B`)
- **L2**: Multiple associations (3-4 pairs, query `C`)
- **L3**: Distractor pairs interleaved
- **L4**: Reordered mapping presentation
- **L5**: Delayed query (filler buffer tokens)
- **L6**: Disjoint identities (train $A\text{--}E/1\text{--}5$, eval $P\text{--}T/6\text{--}0$)
- **L7**: Dynamic variable role assignments
- **L8**: Mixed composite benchmark

## 2. Gating and Promotion Rule
Curriculum advancement requires meeting predefined operational thresholds:
$$\text{Train Acc} \ge 0.80,\quad \text{Val Acc} \ge 0.70,\quad \text{Held-Out Acc} \ge 0.50$$
If any level fails to satisfy these thresholds, curriculum advancement is **immediately halted**.

## 3. Empirical Results (Frozen Baseline & Initial Candidate)

| Level | Name | Train Acc | Val Acc | Held-Out Acc | Threshold Met? | Status |
|---|---|---|---|---|---|---|
| L0 | One Association | 0.0000 | 0.0000 | 0.0000 | False | **HALTED** |
| L1 | Two Associations | N/A | N/A | N/A | N/A | Halted prior to L1 |
| L2 | Multiple Associations | N/A | N/A | N/A | N/A | Halted prior to L2 |
| L3 | Distractors | N/A | N/A | N/A | N/A | Halted prior to L3 |
| L4 | Reordered Mappings | N/A | N/A | N/A | N/A | Halted prior to L4 |
| L5 | Delayed Query | N/A | N/A | N/A | N/A | Halted prior to L5 |
| L6 | Disjoint Identities | N/A | N/A | N/A | N/A | Halted prior to L6 |
| L7 | Randomized Roles | N/A | N/A | N/A | N/A | Halted prior to L7 |
| L8 | Mixed Conditions | N/A | N/A | N/A | N/A | Halted prior to L8 |

## 4. Scientific Verdict
- **Curriculum Halt Boundary**: **Level L0**.
- In the zero-shot unadapted baseline, zero-shot single-association token generation is 0.0000, triggering the immediate halt condition.
- The failure boundary is explicitly recorded at L0; subsequent curriculum levels are marked `UNPROVEN`.
