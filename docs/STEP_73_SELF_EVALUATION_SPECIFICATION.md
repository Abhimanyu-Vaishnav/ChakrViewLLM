# ChakrView Step 73 Specification: Self-Evaluation & Bounded Revision

## 1. Architectural Purpose
Step 73 establishes the authoritative evaluation layer situated between reasoning/proposal generation and patch execution. The [SelfEvaluator](file:///d:/Project/ChakrView/chakrview/cognition/reasoning/evaluation.py) independently audits proposed reasoning and actions against repository context, task constraints, negative boundaries, and security policies.

## 2. Decision Taxonomy
- `ACCEPT`: Fully grounded, internally consistent, zero constraint violations, zero dangerous handles.
- `REVISE`: Resolvable defects detected (e.g. ungrounded claim classified as fact, or preferred alternative hypothesis available).
- `REJECT`: Irrecoverable defects (e.g. hallucinated files/symbols, injected `os.system` / `subprocess` execution handles).
- `ABSTAIN`: Genuine absence of evidence, unresolvable ambiguity, or unresolved contradiction.

## 3. Evaluation Criteria
1. **Factual Grounding**: Confirms all targeted files and symbols exist in the repository context store.
2. **Contradiction Containment**: Ensures zero collisions with negative boundaries or quarantined items.
3. **Provenance Integrity**: Verifies that any assertion claimed as `FACT` carries supporting evidence IDs.
4. **Security Boundary**: Forbids dangerous execution handles (`os.system`, `subprocess`, `eval`, `exec`, `shutil.rmtree`).
5. **Task Constraints**: Asserts compliance with explicit user/system task constraints.

## 4. Bounded Revision Coordinator
- Enforces an explicit limit of maximum 2 revision iterations (`MAX_CYCLES = 2`).
- Prevents infinite self-reflection loops.
- In each cycle, resolvable claims are demoted (e.g. `FACT` without evidence is demoted to `INFERENCE`).
- If defects persist beyond cycle 2, the system fails closed to `REJECT`.
