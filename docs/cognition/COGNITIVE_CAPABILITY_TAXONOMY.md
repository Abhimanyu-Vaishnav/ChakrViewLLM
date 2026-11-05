# COGNITIVE CAPABILITY TAXONOMY (CR-001)

This taxonomy establishes formal definitions, observable behaviors, training signals, and falsification criteria for all 8 cognitive domains in ChakrView.

---

## Domain A: Semantic Understanding
- **Definition**: The internal geometric mapping of concepts such that attributes, category hierarchies, and contextual meanings are structurally organized in embedding and hidden state representations.
- **Observable Behavior**: Correctly categorizing entities, completing analogies ($A:B :: C:D$), identifying synonyms/antonyms without prompt hints.
- **Training Signal**: Contrastive masked language modeling, paired semantic definitions, masked attribute prediction.
- **Benchmark**: WordNet-style relation tests, categorical odd-one-out probes.
- **Failure Mode**: Lexical collision (confusing homonyms or treating surface word co-occurrence as functional equivalence).
- **Memorization vs Generalization Control**: Evaluated on synthetic nonce words with defined attributes (e.g. *"A dax is a small blue mammal. Can a dax swim?"*).

---

## Domain B: Logical Reasoning
- **Definition**: The deductive, inductive, and conditional transformation of truth values across premises according to formal inference rules (Modus Ponens, Modus Tollens, Syllogisms).
- **Observable Behavior**: Given *"All P are Q. X is P."*, outputs *"X is Q."* Given a contradiction, identifies the inconsistent premise.
- **Training Signal**: Formal propositional logic sequences, truth-table derivations, premise-conclusion pairs with negative distractors.
- **Benchmark**: Frozen Propositional Logic Probe Suite (covering Modus Ponens, Syllogisms, Disjunctive Elimination).
- **Failure Mode**: Syllogism belief bias (evaluating validity based on real-world truth rather than logical necessity).
- **Memorization vs Generalization Control**: Abstract symbol logic probes ($A \implies B, \neg B \vdash \neg A$) and counter-factual premises (*"All dogs are reptiles..."*).

---

## Domain C: Mathematical Reasoning
- **Definition**: Symbolic number manipulation, arithmetic execution, order-of-magnitude reasoning, and structured multi-step word problem formulation.
- **Observable Behavior**: Executing multi-digit addition, subtraction, multiplication; calculating rates and ratios; translating natural language word problems into arithmetic equations.
- **Training Signal**: Algorithmic arithmetic traces with scratchpad tokens, number-line relative magnitude embeddings, decomposed step equations.
- **Benchmark**: GSM-Chakr (Grade-school arithmetic with randomized numerical parameters).
- **Failure Mode**: Digit hallucination (generating fluent English text with mathematically impossible intermediate values).
- **Memorization vs Generalization Control**: Large random operand arithmetic ($X + Y = Z$ with $X, Y \in [10^2, 10^5]$) never present in static web crawls.

---

## Domain D: Working Memory & Variable Binding
- **Definition**: The retention of transient variables, entity states, and conversational facts across sequence positions without attention dispersion or forgetting.
- **Observable Behavior**: Retaining user-stated entities across turns ($Turn\ 1: \text{"My name is X"} \to Turn\ 2: \text{"What is my name?"} \to \text{"X"}$).
- **Training Signal**: Key-value retrieval tasks, synthetic variable-binding streams ($x=5, y=x+2, \text{what is } y?$), multi-turn entity tracking.
- **Benchmark**: Needle-In-A-Haystack and multi-turn variable recall probes.
- **Failure Mode**: Context dissolution (attention heads collapsing into generic prompt tokens or default phrase attractors).
- **Memorization vs Generalization Control**: Arbitrary alphanumeric hash/nonce key-value binding probes ($K_1: V_1, K_2: V_2 \implies \text{Lookup}(K_1)$).

---

## Domain E: Multi-Step Problem Solving
- **Definition**: Decomposing a complex goal into sequential intermediate states, solving each subproblem, and integrating partial results into a cohesive final solution.
- **Observable Behavior**: Generating valid, non-redundant subgoals; propagating intermediate solutions forward without losing initial constraints.
- **Training Signal**: Decomposed solution traces, proof graphs, hierarchical task trees.
- **Benchmark**: Multi-step deductive and algorithmic problem suite.
- **Failure Mode**: Premature termination or derailment at step $k$ where an error in step 1 propagates silently into an invalid final output.
- **Memorization vs Generalization Control**: Variable-length reasoning chains where depth $N \in [2, 6]$ is randomly tested on unseen problem graphs.

---

## Domain F: Planning & Constraint Satisfaction
- **Definition**: Organizing actions over an abstract state space to transition from an initial state $S_0$ to goal state $S_g$ subject to invariant constraints $C$.
- **Observable Behavior**: Block-world stack reordering, route planning with blocked nodes, scheduling without conflict.
- **Training Signal**: State-transition graphs, valid execution trajectories, constraint penalty loss.
- **Benchmark**: BlocksWorld, River Crossing, and Resource Scheduling benchmarks.
- **Failure Mode**: Illegal action generation (e.g. moving a block that has another block on top of it).
- **Memorization vs Generalization Control**: Random state permutations where optimal path length requires non-trivial backtracking.

---

## Domain G: Metacognition & Uncertainty Calibration
- **Definition**: Estimating internal probability mass distribution over predictions to accurately distinguish between known facts, assumptions, and insufficient information.
- **Observable Behavior**: Refusing to answer when given incomplete premises; expressing calibrated confidence (low entropy on knowns, high entropy on unknowables).
- **Training Signal**: Abstention supervision, entropy regularization, self-assessment loss.
- **Benchmark**: Calibrated Abstention Suite (unanswerable questions, missing premises, false presuppositions).
- **Failure Mode**: Confabulation / hallucination under epistemic deficit.
- **Memorization vs Generalization Control**: Questions about fictional, non-existent, or future unrecorded events.

---

## Domain H: Self-Correction & Consistency Verification
- **Definition**: The internal capability to evaluate a candidate response against input premises, detect errors or inconsistencies, and output an amended solution.
- **Observable Behavior**: Revising an incorrect intermediate calculation when prompted or during an iterative refinement pass.
- **Training Signal**: Paired error-correction trajectories: $\langle \text{Problem}, \text{Incorrect Candidate}, \text{Critique}, \text{Corrected Solution} \rangle$.
- **Benchmark**: Self-Correction Verification Suite (seeded error correction).
- **Failure Mode**: Sycophancy (changing a correct answer into an incorrect one when challenged).
- **Memorization vs Generalization Control**: Perturbed mathematical and logical problems where candidate answers have injected errors at specific steps.

