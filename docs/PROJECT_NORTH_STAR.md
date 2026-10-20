# CHAKRVIEW — PROJECT NORTH STAR & EXECUTION GUARDRAILS

## 1. Mission
Build ChakrView as an indigenous, modular neural AI system from scratch, progressively developing genuine native language understanding, context grounding, instruction following, reasoning, coding capability, memory, and reliable task execution.

The architecture must remain CPU-first, resource-adaptive, modular, reproducible, and extensible to available accelerators. Never use pretrained external model weights or hide a different model behind a wrapper as proof of ChakrView's native intelligence.

---

## 2. Non-Negotiable Operational Guardrails
1. **Mission-Driven Progress**: Every task must directly advance the long-term ChakrView mission or remove a verified technical blocker.
2. **Honest Demarcation**: Always distinguish native neural capability from deterministic tools, retrieval, heuristics, mocks, and external services.
3. **Reproducible Baselines**: Establish a frozen, reproducible baseline before every meaningful model or runtime change.
4. **Held-Out Functional Evaluation**: Evaluate exclusively on held-out data and functional execution tests; report failures, gaps, and regressions transparently.
5. **No False Intelligence Claims**: Never claim intelligence improvements from lower loss, successful imports, passing smoke tests, or repetition suppression alone.
6. **Minimal Sound Increments**: Prefer the smallest technically justified next step that produces measurable progress. Avoid endless audits, redundant documentation, unnecessary wrappers, and scope drift.
7. **Artifact Protection**: Keep canonical checkpoints, tokenizers, protected runtime files (`chakrview/runtime/session.py`), and established architecture decisions intact unless explicitly authorized.
8. **Authorization Gates**: Never train, fine-tune, alter canonical artifacts, commit, push, merge, tag, or release without explicit user authorization for that specific action.
9. **Empirical Completion**: Do not mark a capability as complete until its acceptance criteria are demonstrably met through reproducible tests.
10. **Living Roadmap**: Maintain a living roadmap tracking capabilities as `IMPLEMENTED`, `TESTED`, `EXPERIMENTAL`, `PLANNED`, or `BLOCKED`.

---

## 3. Decision Hierarchy
$$\text{Genuine model capability} > \text{Reliable evaluation} > \text{Reproducible training} > \text{Useful runtime integration} > \text{Packaging \& presentation}$$

When a proposed task does not advance the mission, challenge it, reduce its scope, or defer it. Keep ChakrView's long-term direction intact across all development sprints.

---

## 4. Standard Sprint Output Requirements
Every execution sprint must report:
1. **Changes made & Mission advancement**: What was actually changed and why it directly advances the mission.
2. **Reproducible evidence**: Exact commands, tests, benchmark artifacts, and output files.
3. **Baseline vs. New measurements**: Direct comparisons, regressions, and documented limitations.
4. **Current blockers**: What remains blocked, unverified, or requiring authorization.
5. **Recommended next step**: Exactly one high-impact next action tied to the living roadmap.
6. **Safety confirmation**: Explicit confirmation of whether training or Git operations occurred.