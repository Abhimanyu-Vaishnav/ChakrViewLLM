# Step 64 Specification: Grounding & Hallucination Containment Gate

## 1. Principle
Neural models and external proposal generators are inherently probabilistic and ungrounded. In ChakrView, **hallucination containment is an architectural invariant, not an afterthought**. The system enforces sovereign deterministic grounding checks before any proposal can be promoted to an executable branch.

---

## 2. Epistemic State Machine
Every proposition and entity in a proposal is classified into an `EpistemicState`:
- `KNOWN`: Empirically confirmed to exist in the current `RepositoryContextStore`.
- `UNKNOWN`: Insufficient evidence or ungrounded proposition; triggers fail-closed `ABSTAIN`.
- `STALE`: Grounded in prior repository state but invalidated by subsequent code changes.
- `CONTRADICTED`: Proven false by live repository inspection (e.g. file or symbol does not exist); triggers immediate `REJECT`.
- `UNAVAILABLE`: Underlying repository component or required test harness is unreachable.

---

## 3. Hallucination Containment Rules

```
           NeuralProposalOutput
                    |
                    v
    [HallucinationContainmentGate]
        |
        +---> Check 1: File Existence in RepositoryContextStore?
        |              No  -> CONTRADICTED -> Immediate REJECT
        |
        +---> Check 2: All Target Files within Task Allowed Scope?
        |              No  -> Scope Violation -> Immediate REJECT
        |
        +---> Check 3: All Target Symbols Exist in AST Signatures?
        |              No  -> Hallucinated Symbol -> Immediate REJECT
        |
        +---> Check 4: Preconditions Verifiable in RepositoryState?
        |              No  -> UNKNOWN / Unverifiable -> ABSTAIN
        |
        v
       All Checks Passed
        |
        v
  [Promote to SynthesizedCandidate]
```

### Deterministic Grounding Tests
1. **File Existence**: Every target file and patch path is looked up in `RepositoryContextStore._file_cache`. Any missing path is labeled an unfounded/hallucinated file.
2. **Symbol Existence**: Every referenced function or class name is verified against `ModuleInspection.functions` and `ModuleInspection.classes`. Nonexistent symbols are rejected as hallucinations.
3. **Scope Enclosure**: All candidate files must be a subset of task `allowed_modified_files`.
