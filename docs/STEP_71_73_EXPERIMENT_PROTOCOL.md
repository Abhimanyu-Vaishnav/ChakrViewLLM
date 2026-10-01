# ChakrView Steps 71–73 Experiment Protocol

## 1. Objective
Demonstrate an integrated cognitive loop where a task flows from cognitive context through neural proposal generation, structured reasoning, critical analysis, alternative hypothesis formulation, and independent self-evaluation with bounded revision, preserving the neural invariant ($\Delta W = 0$).

## 2. Experimental Pipeline
```
USER TASK
   ↓
CognitiveContextBundle (Step 67)
   ↓
ChakrMicro Neural Proposal (Step 68: ProposalContract)
   ↓
StructuredReasoningEngine (Step 71: StructuredReasoningArtifact)
   ↓
CriticalThinkingEngine (Step 72: CriticalAnalysisReport + AlternativeHypotheses)
   ↓
SelfEvaluator (Step 73: SelfEvaluationReport)
   ↓
BoundedRevisionCoordinator (Cycles <= 2)
   ↓
Final Authoritative Verdict (ACCEPT / REVISE / REJECT / ABSTAIN)
```

## 3. Mandatory Experimental Controls
- Control 1: Structured reasoning creation (claims, assumptions, observations).
- Control 2: Fact provenance preservation (every FACT has evidence ID).
- Control 3: Critical thinking evidence balance evaluation.
- Control 4: Formulation and evaluation of competing alternative hypotheses.
- Control 5: Self-evaluation acceptance of valid grounded proposal.
- Control 6: Rejection of hallucinated file claims.
- Control 7: Rejection of prohibited execution handles (`os.system`).
- Control 8: Abstention on quarantined contradictions.
- Control 9: Abstention / UNKNOWN representation on empty evidence context.
- Control 10: Successful resolution of flawed claims via bounded revision.
- Control 11: Termination of bounded revision cycles within 2 iterations.
- Control 12: Absolute neural core immutability ($\Delta W = 0$).
