# Step 67: Experiment Protocol

## 1. Primary Scientific Question
Can ChakrView deterministically compose a unified cognitive context from:
1. current repository-grounded evidence,
2. valid recalled episodic memories,
3. verified negative-boundary memories,
4. current task constraints,
and expose that context to the passive `NeuralProposalAdapter` while preserving provenance, deterministic ordering, conflict isolation, budget limits, and complete authority separation?

## 2. Hypothesis
A dedicated composition layer (`UnifiedCognitiveContextComposer`) can aggregate multi-source grounded evidence and episodic recall, systematically quarantine conflicts and negative constraints into explicit channels, and yield a bit-for-bit deterministic, passive data payload that completely insulates the repository and memory indexes from unauthorized mutation.

## 3. Experimental Protocol & Conditions (A–N)

| Condition | Description | Verification Criterion |
|---|---|---|
| **A** | Current repository evidence included | `positive_evidence` contains AST & graph evidence with `KNOWN` epistemic state |
| **B** | Valid episodic memory included | Recallable positive memory is present in `positive_memories` with `ACTIVE` status |
| **C** | Negative boundary isolated | Negative failure pattern is isolated to `negative_boundaries` and excluded from positives |
| **D** | Stale memory excluded | Memories targeting drifted or deleted modules are excluded from positives |
| **E** | Superseded memory excluded | Memories flagged `superseded_by` are excluded from positives |
| **F** | Conflicted memory quarantined | Collisions with negative boundaries or task constraints are marked `CONFLICTED` |
| **G** | ABSTAIN state produces no positive memory | Fingerprint mismatch triggers `abstained=True` and 0 positive items |
| **H** | 100% Provenance Coverage | All composed items have valid non-empty provenance fields |
| **I** | Budget limits obeyed | Item counts strictly respect `CognitiveContextBudget` limits |
| **J** | Deterministic repeatability | Identical inputs produce bit-for-bit identical context and fingerprint |
| **K** | Passive neural context | `to_neural_context()` contains passive, non-callable, read-only data |
| **L** | Zero memory mutation | Neural adapter execution does not alter or append to `RepositoryMemoryIndex` |
| **M** | Neural baseline bit-exact | Pre- and post-run model hash equals `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` |
| **N** | Step 59–66 compatibility | Standalone retrievers and previous step pipelines continue passing without regressions |

## 4. Execution Command
```powershell
.venv\Scripts\python.exe scripts/experiment_step67_cognitive_context.py
```
