# ChakrView Step 57: Canonical Experience Format

- **Document Version**: 1.0.0
- **Status**: Ratified Format Specification
- **Subject**: Structured Experiential Record for Episodic Memory and RIL

---

## 1. Schema Definition

An **ExperienceRecord** captures the essential causal pattern of an episode so that future attempts on the same or related tasks can retrieve and condition on verified knowledge:

```text
ExperienceRecord
├── experience_id: str ("exp_cog_{task_id}_{timestamp}")
├── task_id: str
├── task_family: str ("syntax_repair", "arithmetic", "state_transition", "function_repair")
├── task_description: str
├── context: Dict[str, Any]
├── attempt_count: int
├── initial_action: str
├── failure_observation: Optional[str]
├── diagnosis: Optional[str]
├── correction: Optional[str]
├── verified_result: str ("SUCCESS" | "FAILURE")
├── what_worked: str
├── what_failed: str
├── reusable_pattern: str
├── verified: bool
└── timestamp_utc: str
```

---

## 2. Invariants & Guardrails

1. **Verification Requirement**:
   Only episodes with `verified_result == "SUCCESS"` and objective test pass in ChakrKshetra can have `verified: True`.
2. **No Inferred Success**:
   If an episode failed or timed out, `verified: False`, and it must **never** be used to train adapters.
3. **Structured Lesson Extraction**:
   `what_worked` and `what_failed` must be extracted from actual test assertion diffs and traceback diagnostics, never hallucinated.
4. **Token Bound**:
   `reusable_pattern` is strictly capped at $\le 100$ characters so it fits effortlessly inside the `<MEMORY>` block of the `<CORTEX_CONTEXT>` contract.
