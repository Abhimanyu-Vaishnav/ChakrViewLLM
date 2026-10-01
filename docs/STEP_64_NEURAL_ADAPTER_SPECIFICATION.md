# Step 64 Specification: Neural Proposal Adapter & Authority Boundaries

## 1. Authority Separation Contract
In the ChakrView cognitive architecture:

$$\textbf{Neural Model} \longrightarrow \text{Proposes, synthesizes, and prioritizes candidate hypotheses}$$
$$\textbf{Deterministic System} \longrightarrow \text{Validates facts, verifies boundaries, executes, tests, rolls back, and commits}$$

Under no circumstances does model generation grant ambient authority or bypass verification.

---

## 2. Pluggable Adapter Interface
The `NeuralProposalAdapter` defines a model-agnostic contract:

```python
class NeuralProposalAdapter(ABC):
    @abstractmethod
    def generate_proposal(
        self,
        context: GroundedContextBundle,
    ) -> Optional[NeuralProposalOutput]:
        """Propose a refactoring candidate from grounded context."""
        pass
```

### Supported Implementations:
- `MockNeuralProposalAdapter`: Deterministic reference adapter used for unit testing, CI pipelines, and benchmarking grounding boundaries without external network dependencies.
- Pluggable future adapters (e.g. local Qwen, cloud LLMs, or indigenous ChakrMicro fine-tunes).

---

## 3. Proposal Pipeline & Safety Gate
The neural proposal flow proceeds as follows:
1. `GroundedContextRetriever` constructs a bounded `GroundedContextBundle`.
2. `NeuralProposalAdapter` generates `NeuralProposalOutput`.
3. `HallucinationContainmentGate` verifies symbol and file grounding.
4. `CandidateNormalizer` ensures structural validity and patch formatting.
5. `DeterministicSafetyGate` evaluates scope boundaries, depth ceilings, and memory validity.
6. `SynthesisAwareBranchingCoordinator` executes the proposal inside an `IsolatedWorkspace`.
7. `RepositoryVerifier` runs 4 tiers of automated test verification.
8. If any failure occurs, `RepositoryPatchCoordinator` executes atomic rollback and confirms bit-exact fingerprint restoration.
