# ChakrView Step 57: Learning Episode Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Specification
- **Subject**: Canonical LearningEpisode and Attempt Abstractions

---

## 1. Foundational Architecture

A **LearningEpisode** models a discrete, bounded cognitive problem-solving trajectory. It captures every attempt, observation from ChakrKshetra, diagnostic finding, corrective retry, verification, and subsequent learning event.

```text
LearningEpisode
├── episode_id: str
├── task_id: str
├── task_spec: Dict[str, Any]
├── initial_context: Dict[str, Any]
├── attempts: List[Attempt]
├── executed_stages: List[str]
├── final_result: str ("SUCCESS" | "FAILURE")
├── learning_status: str ("NEW_LEARNING" | "REPLAY_REUSE" | "NO_CHANGE")
├── experience_record: Optional[Dict[str, Any]]
├── candidate_update: Optional[Dict[str, Any]]
├── verification_result: Dict[str, Any]
└── promotion_status: str ("NOT_PROMOTED" | "PROMOTED" | "QUARANTINED")
```

---

## 2. The Attempt Schema

Each turn within an episode is an **Attempt**:
```text
Attempt
├── attempt_id: int
├── state: str
├── action: str (proposed code or completion)
├── observation: str (raw stderr / test outcome from ChakrKshetra)
├── evaluation: str ("PASS" | "FAIL")
├── diagnosis: Optional[str] (identified root cause)
├── correction: Optional[str] (applied corrective patch)
└── duration_seconds: float
```

---

## 3. Supported Cognitive Stages

The orchestration loop records only stages that actually transpired:
1. `UNDERSTAND`: Parse requirements and constraints.
2. `PLAN`: Decompose task into sequential actions.
3. `ACT`: Neural generation of initial candidate.
4. `OBSERVE`: External execution and consequence capture in ChakrKshetra.
5. `EVALUATE`: Determine pass/fail based on objective test results.
6. `DIAGNOSE`: Classify error type and localize defect if evaluation failed.
7. `CORRECT`: Formulate corrective patch.
8. `RETRY`: Execute retry in ChakrKshetra.
9. `VERIFY`: Confirm all assertions pass.
10. `REMEMBER`: Extract verified experience and store into EpisodicMemoryStore.
11. `LEARN`: Update candidate task adapter or context conditioning for future reuse.

---

## 4. Invariants

1. **Deterministic Serialization**: Must serialize cleanly to UTF-8 JSON.
2. **No Falsification**: Only actual executed stages and observations may be recorded.
3. **Bounded Iterations**: Attempts are strictly capped (default $\le 3$ attempts) to prevent infinite loops.
