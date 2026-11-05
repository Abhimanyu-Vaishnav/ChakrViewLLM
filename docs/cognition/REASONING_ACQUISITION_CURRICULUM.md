# REASONING ACQUISITION CURRICULUM (CR-001)

A systematic, 8-level progressive curriculum designed to transition ChakrView from raw token continuation into structured, verifiable cognitive inference.

---

## 1. Curriculum Progression Structure

```
Level 8: Metacognitive Calibration & Self-Correction (Uncertainty & Refinement)
   ↑
Level 7: Planning & Goal Decomposition (Subgoal Sequencing & Constraint Resolution)
   ↑
Level 6: Multi-Step Mathematical & Deductive Problems (Multi-hop Composition)
   ↑
Level 5: Causal Modeling & Analogical Transfer (If-Then-Because & Structural Mapping)
   ↑
Level 4: Formal Deductive & Inductive Reasoning (Syllogisms & Pattern Generalization)
   ↑
Level 3: Algorithmic Arithmetic & Numerical Foundations (Scratchpad Addition/Multiplication)
   ↑
Level 2: Factual Inference & Entity Attribute Binding (Key-Value Induction Circuits)
   ↑
Level 1: Semantic Relations & Concept Distinctions (Taxonomies & Opposites)
```

---

## 2. Level-by-Level Specification

### Level 1: Semantic Relations & Concept Distinctions
- **Task**: Identify hypernyms, hyponyms, antonyms, functional properties.
- **Example**:
  - `Prompt`: `Concept: Bicycle. Primary function: Transportation. Number of wheels: 2. Propelled by: Pedals. Question: How is a bicycle propelled?\nResponse:`
  - `Target`: `A bicycle is propelled by pedals.`
- **Training Objective**: Masked Causal Cross-Entropy over response tokens only.
- **Quality Gate**: $\ge 90\%$ accuracy on unseen entity-attribute queries.

### Level 2: Factual Inference & Entity Attribute Binding
- **Task**: Exact variable binding and key-value retrieval across turn boundaries.
- **Example**:
  - `Prompt`: `User: Server alpha is deployed at port 8080. Server beta is deployed at port 9090.\nUser: What port is server alpha using?\nAssistant:`
  - `Target`: `Server alpha is using port 8080.`
- **Failure Mode**: Attending to the most recent port (9090) instead of binding `alpha -> 8080`.
- **Quality Gate**: $\ge 85\%$ retrieval accuracy across 200 held-out randomized bindings.

### Level 3: Algorithmic Arithmetic & Numerical Foundations
- **Task**: Multi-digit addition, subtraction, and comparison using scratchpad tokenization.
- **Example**:
  - `Prompt`: `Calculate 347 + 582 step by step.\nAssistant:`
  - `Target`: `[Step 1] Units: 7 + 2 = 9. [Step 2] Tens: 4 + 8 = 12 (carry 1). [Step 3] Hundreds: 3 + 5 + 1 = 9. [Result] 929.`
- **Training Signal**: Direct supervision on decomposed intermediate carry operations.
- **Quality Gate**: $\ge 80\%$ accuracy on random 3-digit additions.

### Level 4: Formal Deductive & Inductive Reasoning
- **Task**: Modus Ponens, Modus Tollens, Disjunctive Syllogism on abstract and real symbols.
- **Example**:
  - `Prompt`: `Premise 1: If an engine overheats, it triggers an alarm. Premise 2: The engine overheated. Conclusion:\nAssistant:`
  - `Target`: `The alarm is triggered.`
- **Quality Gate**: $\ge 75\%$ logical validity across 100 formal deduction probes.

### Level 5: Causal Modeling & Analogical Transfer
- **Task**: Explaining *why* an event occurred and mapping relationships across domains.
- **Example**:
  - `Prompt`: `Why did the puddle evaporate when the sun came out?\nAssistant:`
  - `Target`: `Solar radiation increased the temperature of the water, providing thermal energy that transformed liquid water into water vapor.`
- **Quality Gate**: Coherent causal attribution without temporal reversals.

### Level 6: Multi-Step Word Problems
- **Task**: Multi-hop GSM8K-style problems requiring variable extraction, equation setup, and sequential solution.
- **Example**:
  - `Prompt`: `A library has 120 books. 30 are borrowed on Monday. Half of the remainder are borrowed on Tuesday. How many books remain in the library?\nAssistant:`
  - `Target`: `[Step 1] Monday remainder: 120 - 30 = 90. [Step 2] Tuesday borrowed: 90 / 2 = 45. [Step 3] Tuesday remainder: 90 - 45 = 45. [Final] 45 books remain.`
- **Quality Gate**: $\ge 60\%$ multi-step solution accuracy.

### Level 7: Planning & Goal Decomposition
- **Task**: Generating valid, sequential action steps to achieve a specified goal under constraints.
- **Example**:
  - `Prompt`: `Goal: Deploy a containerized web application. Constraints: Database must start before web server. Provide the action plan:\nAssistant:`
  - `Target`: `1. Initialize configuration environment. 2. Launch database container and verify health. 3. Start web server container linked to active database. 4. Verify endpoint response.`
- **Quality Gate**: 0% constraint violations in generated plans.

### Level 8: Metacognitive Calibration & Self-Correction
- **Task**: Recognizing unanswerable queries, detecting planted contradictions, and repairing flawed calculations.
- **Example**:
  - `Prompt`: `User: Who was the president of the United States in the year 2350?\nAssistant:`
  - `Target`: `That question cannot be answered because the year 2350 is in the future and unrecorded.`
- **Quality Gate**: $\ge 90\%$ abstention accuracy on epistemically invalid queries.

