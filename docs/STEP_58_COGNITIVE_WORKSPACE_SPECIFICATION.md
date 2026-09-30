# ChakrView Step 58: Cognitive Workspace Architecture Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Specification (Step 58)
- **Scope**: Reusable Multi-Turn Cognitive Orchestration Layer for ChakrMicro

---

## 1. System Philosophy

The **Cognitive Workspace** is the runtime orchestration layer that governs ChakrMicro's interaction with the external environment (**ChakrKshetra**), coordinates multi-turn attempt loops, tracks working cognitive state, and interfaces with the multi-tier memory architecture.

The workspace operates under the strict invariant:
```
CHAKRVIEW NEURAL CORE ≠ CHAKRKSHETRA ≠ WORKING MEMORY ≠ EPISODIC MEMORY ≠ SEMANTIC MEMORY ≠ RIL ≠ ADAPTER ≠ EVALUATOR ≠ HOST APPLICATION
```

The workspace itself is **NOT** the neural brain. It is the deterministic cognitive scaffolding that:
1. Translates task requirements into structured working states.
2. Dispatches actions to ChakrKshetra.
3. Captures structured observations and synthesizes diagnoses.
4. Retrieves relevant memories from episodic and semantic memory stores.
5. Injects bounded cognitive context into ChakrMicro's conditioning horizon.
6. Records complete, auditable trajectories.
7. Dispatches verified episodes to the memory consolidation pipeline.

---

## 2. The 11-Stage Cognitive Loop

The canonical cognitive loop implemented by `CognitiveWorkspace` is:

```
UNDERSTAND
   ↓
 PLAN
   ↓
  ACT
   ↓
OBSERVE
   ↓
DIAGNOSE
   ↓
CORRECT
   ↓
 VERIFY
   ↓
REMEMBER
   ↓
CONSOLIDATE
   ↓
 REUSE
   ↓
REFLECT
```

### Stage Definitions

1. **UNDERSTAND**: Parses task specifications, identifies constraints, determines task family, and initializes working memory.
2. **PLAN**: Establishes target checkpoints and strategy (initial generation vs repair vs state transition).
3. **ACT**: Emits an executable code or system mutation from the model (or candidate generator) under the current cognitive context.
4. **OBSERVE**: Executes the action within ChakrKshetra's isolated sandbox; captures stdout, stderr, exit code, and test assertions.
5. **DIAGNOSE**: Evaluates observations against pass/fail criteria; on failure, extracts traceback, fault line, and failure reason.
6. **CORRECT**: Synthesizes a corrective action using the diagnosis and relevant memories without exceeding the context horizon.
7. **VERIFY**: Confirms all objective tests and assertions pass cleanly in ChakrKshetra.
8. **REMEMBER**: Records the completed attempt history into an auditable `LearningEpisode` and creates an `ExperienceRecord`.
9. **CONSOLIDATE**: Routes verified experiences to the `MemoryConsolidator` to update or instantiate generalized semantic patterns.
10. **REUSE**: Exposes consolidated patterns and verified episodes to subsequent turns or new tasks via explainable ranking.
11. **REFLECT**: Executes a structured reflection analyzing what failed, what worked, why it worked, transfer boundaries, and non-generalizable limits.

---

## 3. Component Architecture & Data Flow

```
                 ┌────────────────────────────────┐
                 │       Task Specification       │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │       CognitiveWorkspace       │
                 ├────────────────────────────────┤
                 │ - Working Memory (State)       │
                 │ - Trajectory Logger            │
                 │ - Turn & Attempt Controller    │
                 └──────┬───────────────┬─────────┘
                        │               │
     Context Injection  │               │ Action Execution
                        ▼               ▼
            ┌─────────────────┐   ┌─────────────────┐
            │ ChakrMicro Core │   │  ChakrKshetra   │
            │ (Frozen Weights)│   │ (Execution Env) │
            └─────────────────┘   └────────┬────────┘
                                           │
                               Observation │
                                           ▼
                                  ┌─────────────────┐
                                  │ Arena Evaluator │
                                  └────────┬────────┘
                                           │
                                           ▼
                   ┌──────────────────────────────────────┐
                   │ Structured Diagnosis / Verification  │
                   └──────────────────┬───────────────────┘
                                      │
                         Verified Exp │
                                      ▼
                        ┌────────────────────────────┐
                        │   Multi-Tier Memory        │
                        ├────────────────────────────┤
                        │ - Episodic Memory Store    │
                        │ - Semantic Consolidator    │
                        │ - Explainable Retriever    │
                        └────────────────────────────┘
```

---

## 4. Working Cognitive State Schema

Working memory is captured in a strictly typed, serializable dataclass `CognitiveWorkingState`:

- `task_id`: Unique identifier of the current task.
- `task_family`: Broad categorization (e.g., `function_repair`, `state_machine`, `syntax_correction`).
- `objective`: Natural language description of the goal.
- `current_state`: Operational phase (`INITIALIZING`, `PLANNING`, `EXECUTING`, `EVALUATING`, `DIAGNOSING`, `REPAIRING`, `VERIFIED`, `FAILED`).
- `current_plan`: Current tactical plan steps.
- `current_action`: Latest action emitted.
- `last_observation`: Most recent observation from ChakrKshetra.
- `diagnosis`: Diagnostic summary of the failure (if any).
- `attempt_index`: Current attempt (1-based).
- `max_attempts`: Maximum allowed attempts before marking failure.
- `completed`: Boolean flag indicating completion of execution loop.
- `verified`: Boolean flag indicating verified success in ChakrKshetra.
- `failure_history`: List of previous failed attempts with observations and diagnoses.
- `relevant_memories`: List of memory IDs or summary records retrieved for this turn.
- `active_adapter`: Identifier of mounted native task adapter (if any).
- `timestamp_utc`: Timestamp of creation.

---

## 5. Action / Observation Trajectory Compatibility

Every turn in `CognitiveWorkspace` produces an auditable entry conforming to the Step 54 format:

```text
<TRAJECTORY>
<SPEC>
{task_id}: {objective}
</SPEC>
<STATE>
Attempt {attempt_index} / {max_attempts}
</STATE>
<ACTION>
{current_action}
</ACTION>
<OBSERVATION>
{observation}
</OBSERVATION>
<DIAGNOSIS>
{diagnosis or "None"}
</DIAGNOSIS>
<NEXT_ACTION>
{correction or "Terminal Verification"}
</NEXT_ACTION>
<RESULT>
{SUCCESS or FAILURE}
</RESULT>
</TRAJECTORY>
```

This ensures full mathematical auditability without hidden state transitions.
