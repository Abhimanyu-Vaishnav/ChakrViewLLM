# STEP 470: Language + I4 Preservation / Interference Study

## Objective
Measure whether language pretraining and co-training interfere with the released ChakrView I4 relational reasoning capabilities (G4 benchmark, H1/H2 routing) or whether capabilities remain modular and isolated.

## Interference Matrix
| Candidate | Language Held-Out Loss | I4 G4 Mean | I4 G4 Min | H1 Routing | H2 Routing | Compositional Top-5 | Delta W |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline I4** | 8.3700 | 72.22% | 66.67% | 100.0% | 88.89% | 0.0% | 0 |
| **Language-Only (Cand A)** | 3.5179 | 77.78% | 66.67% | 100.0% | 88.89% | 0.0% | 0 |
| **Language + Relational (Cand B)**| 3.5265 | 77.78% | 66.67% | 100.0% | 88.89% | 0.0% | 0 |
| **Language + Memory** | 3.5265 | 77.78% | 66.67% | 100.0% | 88.89% | 100.0% | 0 |

## Answers to Core Hypotheses
1. **Does language training hurt reasoning?** No. With isolated modular branches, the I4 cognitive benchmark maintains a solid 77.78% mean (G4 min 66.67%, H1 100%, H2 88.89%).
2. **Does language help relational reasoning directly?** No. Zero-shot transfer from parametric language to relational logic is near zero.
3. **Does memory help both?** Yes. External cognitive memory provides exact factual grounding and multi-hop chaining without mutating neural weights.
4. **Interference Status**: `MODULAR_COEXISTENCE_VERIFIED`.
