# Step 63 Specification: Autonomous Multi-Branch Strategy Synthesis & Safety Gating

## 1. System Overview
Step 63 introduces autonomous candidate strategy synthesis into the ChakrView repository cognition layer. While Step 62 demonstrated observation-driven branching and recovery over statically declared branches, Step 63 enables the dynamic synthesis and recombination of refactoring candidates from past execution traces, failed branch observations, and semantic memory patterns—without sacrificing deterministic arbitration or fail-closed safety.

```
+-------------------------------------------------------------+
|               Strategy Generation Layer                     |
|  (Trace Recombination / Semantic Memory / Future LLM)      |
+-------------------------------------------------------------+
                              |
                              v [SynthesizedCandidate]
+-------------------------------------------------------------+
|                  Candidate Normalization                     |
|      (Schema validation, scope targets, step ordering)      |
+-------------------------------------------------------------+
                              |
                              v [Normalized Proposals]
+-------------------------------------------------------------+
|                 Candidate Deduplication                     |
|           (Fingerprint hashing, provenance merging)         |
+-------------------------------------------------------------+
                              |
                              v [Unique Candidates]
+-------------------------------------------------------------+
|                 Deterministic Safety Gate                   |
|     (Allowed files, domain match, memory revalidation)      |
+-------------------------------------------------------------+
           |                                  |
    [REJECT / ABSTAIN]                    [ACCEPT]
           |                                  |
           v                                  v
      Fail Closed             ObservationDrivenBranchingCoordinator
                                              |
                                              v
                                      IsolatedWorkspace
                                              |
                                              v
                                   Four-Tier Verification
```

---

## 2. Core Abstractions

### 2.1 `SynthesizedCandidate`
A passive, serializable proposal representing an intended refactoring strategy:
- `candidate_id`: Deterministic unique identifier.
- `objective`: High-level goal.
- `ordered_steps`: Sequence of discrete `RefactoringStep` instances.
- `allowed_files`: Explicit boundary set for modifications.
- `preconditions`: Required state assertions prior to execution.
- `expected_observations`: Target intermediate observation states.
- `failure_conditions`: Early-exit triggering predicates.
- `recovery_policy`: Transition policy upon verification failure.
- `provenance`: Audit object recording origins, source memories, and domain tags.
- `fingerprint`: Deterministic SHA-256 hash of objective, allowed files, step patches, and preconditions.

### 2.2 `CandidateNormalizer`
Validates structural integrity:
- Enforces non-empty identifiers, objectives, and steps.
- Asserts that all patch target files reside strictly within candidate `allowed_files`.
- Rejects duplicate step IDs.
- Validates recovery policies against allowed enums.

### 2.3 `DeterministicSafetyGate`
A sovereign gate evaluating:
- **Scope Compliance**: Verifies that candidate files are a strict subset of task `allowed_modified_files`.
- **Operational Ceilings**: Enforces `max_candidate_steps` (default: 6) and `max_recombination_depth` (default: 3).
- **Negative-Transfer Defense**: Detects mismatches between task domain and candidate domain tags, triggering deterministic `ABSTAIN`.
- **Memory Revalidation**: Queries `RepositoryImpactAnalyzer` using reference baseline states; any candidate depending on `STALE` or `INVALID` memories is blocked.

### 2.4 `CandidateDeduplicator`
- Hashing candidate content into SHA-256 fingerprints.
- Identical strategies are collapsed into a canonical instance while union-merging provenance metadata (branch IDs, memory IDs, domain tags).

### 2.5 `StrategyRecombiner`
- Combines successful steps from partial traces (e.g. prefix steps from one branch, suffix steps from another).
- Confirms end-to-end file boundary compliance before generating a recombined candidate.

---

## 3. Execution Authority Separation
The system strictly enforces that:
1. Generation layers possess **zero execution authority**.
2. Proposals are converted to `RefactoringBranch` only after passing normalization, deduplication, and safety gating.
3. Synthesized branches are assigned lower baseline priority than authored branches, guaranteeing that empirical alternatives are attempted only when authored strategies fail or when explicit synthesis is requested.
