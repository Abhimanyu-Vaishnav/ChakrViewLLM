# CHAKRVIEW CAPABILITY REGISTRY

The central authoritative registry tracking all cognitive, linguistic, and operational capabilities in ChakrView.

---

## 1. Status Classifications

- **NOT DEMONSTRATED**: Model fails completely or produces random/hallucinatory output on held-out probes.
- **UNDER EXPERIMENT**: Formal experimental design active; isolated training or evaluation underway.
- **PARTIALLY VALIDATED**: Measurable improvement demonstrated on specific subsets, but lacks full generalization or exhibits positional/distractor sensitivity.
- **VALIDATED**: Statistically robust, reproducible performance verified across out-of-distribution held-out benchmarks without regression on previously acquired capabilities.

---

## 2. Capability Ledger

| ID | Capability Domain | Current Evidence | Known Failures & Bottlenecks | Required Benchmark Suite | Baseline Checkpoint | Candidate Under Test | Success Criteria | Regression Criteria | Current Status |
|---|---|---|---|---|---|---|---|---|---|
| **CAP-01** | **Language Fluency & Syntax** | Grammatically fluent English in greetings & neural definitions. Top-5 syntactic probe accuracy 15.0%. | Hindi perplexity remains high (>5,000) due to byte fragmentation and undertraining. | Frozen 40-Probe Cloze Benchmark (`step102_syntactic_probes.json`) | `checkpoint_0000500.pt` | `sft_checkpoint_step_60.pt` | Top-1 accuracy $\ge 50\%$, English PPL $< 80.0$ | English PPL increase $> 10\%$ | **PARTIALLY VALIDATED** |
| **CAP-02** | **Conversational Identity & Coherence** | Consistently responds with name `"ChakrView"`, avoids repetition loops under rep penalty 1.15. | Single-turn greetings occasionally trigger generic assistance formulas. | Universal Capability Benchmark (Identity & Coherence probes) | `checkpoint_0000500.pt` | `sft_checkpoint_step_60.pt` | 100% exact match on identity; zero loops under greedy decoding | Loss of identity stability | **VALIDATED** |
| **CAP-03** | **In-Context Variable Binding** | Identifies query prefix (`"Your name is..."`), but hallucinates subwords on novel entities. | Fails to bind novel entities ($EM=0\%$ on out-of-distribution proper nouns). | Synthetic Disjoint Key-Value Probe Suite (`CR-EXP-01`) | `sft_checkpoint_step_200.pt` | `CR-EXP-01 Candidate` | Novel Association EM $\ge 60\%$, Distractor resistance $\ge 50\%$ | Loss of general conversational syntax | **UNDER EXPERIMENT** |
| **CAP-04** | **Multi-Turn Working Memory** | Can maintain 1 turn of context when strictly filtered; 0% accurate on variable recall. | Prior turns dilute attention weights, causing context dissolution. | Multi-turn Key-Value Memory Probes (2 to 4 turn dialogues) | `sft_checkpoint_step_200.pt` | `CR-EXP-01 Candidate` | 2-turn entity recall $\ge 70\%$ | Degradation to single-turn responses | **NOT DEMONSTRATED** |
| **CAP-05** | **Logical Reasoning (Deduction)** | Scores 0% on abstract syllogisms; answers based on surface text similarity. | Cannot apply Modus Ponens/Tollens over novel symbols ($A \implies B$). | Frozen Propositional Logic Probe Suite (50 probes) | `checkpoint_0000500.pt` | Planned (CR-002) | Syllogistic validity $\ge 75\%$ | Generation of self-contradictory text | **NOT DEMONSTRATED** |
| **CAP-06** | **Mathematical Reasoning (Arithmetic)** | Emits random numeric tokens (`"15 + 27 = 12 7"`). | No internalized addition or place-value carry circuits. | GSM-Chakr & Random Operand Arithmetic Probes | `checkpoint_0000500.pt` | Planned (CR-003) | 2-digit addition $EM \ge 80\%$ | Digit hallucination rate $> 20\%$ | **NOT DEMONSTRATED** |
| **CAP-07** | **Multi-Step Problem Solving** | Derails after step 1; error propagates unchecked into final output. | Cannot compose intermediate equations sequentially. | Multi-hop Deductive & Math Problem Suite (60 probes) | `checkpoint_0000500.pt` | Planned (CR-003) | 3-hop valid execution $\ge 50\%$ | Premature stopping or loop trapping | **NOT DEMONSTRATED** |
| **CAP-08** | **Planning & Constraints** | Generates unconstrained text violating dependency order. | Cannot verify state invariants or track blocked states. | BlocksWorld & Scheduling Constraint Suite | `checkpoint_0000500.pt` | Planned (CR-004) | Constraint satisfaction $\ge 65\%$ | Illegal action rate $> 10\%$ | **NOT DEMONSTRATED** |
| **CAP-09** | **Uncertainty Calibration & Abstention** | Eradicated toxic `"context"` phrase sink in CBR-SFT-08, but still confabulates on unknowables. | Cannot reliably detect epistemic boundaries on unanswerable queries. | Calibrated Abstention Benchmark (50 unanswerable probes) | `sft_checkpoint_step_60.pt` | Planned (CR-EXP-02) | Abstention accuracy $\ge 85\%$, false refusal $< 10\%$ | Over-refusal on answerable queries | **PARTIALLY VALIDATED** |
| **CAP-10** | **Self-Correction & Error Detection** | Cannot detect errors in its own outputs; sycophantically changes answers when questioned. | Lacks internal critique and verification mechanisms. | Seeded Error Correction & Consistency Suite | `checkpoint_0000500.pt` | Planned (CR-005) | Error repair rate $\ge 55\%$ | Corrupting previously correct answers | **NOT DEMONSTRATED** |

