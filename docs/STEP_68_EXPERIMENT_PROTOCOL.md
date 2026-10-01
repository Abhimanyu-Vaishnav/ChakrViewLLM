# Step 68: Experiment Protocol

## 1. Primary Scientific Question
"Can ChakrView's frozen neural core consume the unified cognitive context produced by Step 67 and produce a deterministic, provenance-grounded proposal, while remaining strictly proposal-only and retaining zero execution, mutation, memory-write, or repository-write authority?"

## 2. Hypothesis
By feeding the passive `CognitiveContextBundle` through a deterministic input encoder (`NeuralProposalInputEncoder`), executing inference via the frozen `ChakrMicro` core in `InferenceEngine`, and gating the resulting `ProposalContract` through an authoritative `ProposalValidator`, ChakrView can generate verifiable, provenance-grounded proposals while guaranteeing `ΔW = 0`, zero repository mutations, and zero memory corruption.

## 3. Experimental Controls

| Control | Scenario | Verification Criterion |
|---|---|---|
| **Primary Pipeline** | Full pipeline on known refactoring task (discount invariant) | Proposal reaches `ACCEPTED_FOR_EXECUTION_REVIEW` with patch `min(order_total, discount)` |
| **Control A** | Full grounded cognitive context | Confidence >= 0.85, multiple structured claims across epistemic partitions |
| **Control B** | Episodic memory removed from context | Proposal succeeds with lowered confidence based on grounded evidence alone |
| **Control C** | Target file removed from repository store | Proposal targets missing file and is authoritatively `REJECTED` |
| **Control D** | Conflicting negative boundary injected | Negative boundary isolated and checked; violating patch rejected |
| **Control E** | Repeated inference with identical inputs | Identical proposal ID, patches, validation status, and generated tokens |
| **Immutability** | Pre- and post-run parameter hash calculation | Exact match with `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` |

## 4. Execution Command
```powershell
.venv\Scripts\python.exe scripts/experiment_step68_neural_proposal.py
```
