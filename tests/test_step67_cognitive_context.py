"""
Tests for Step 67: Unified Cognitive Context Composition.

Validates:
1. Composition of current repository evidence.
2. Inclusion of valid recalled episodic memories.
3. Negative boundary isolation in explicit negative channel.
4. Exclusion of stale memories from positive context.
5. Exclusion of superseded memories from positive context.
6. Conflict arbitration: conflicting positive memories marked CONFLICTED and quarantined.
7. Explicit task constraints conflict detection.
8. Nonexistent repository module conflict detection.
9. Abstention handling when recall coordinator abstains.
10. Repository fingerprint mismatch abstention.
11. 100% provenance tracking for all included context items.
12. Budget enforcement for evidence, memories, negative boundaries, and total items.
13. Deterministic bit-for-bit repeatability across repeated compositions.
14. to_neural_context produces strictly passive data without mutation capabilities or handles.
15. NeuralProposalAdapter consumes to_neural_context successfully and generates proposal.
16. Neural baseline parameter count and weight hash verification.
"""

from __future__ import annotations

import hashlib
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.episodic_recall import (
    EpisodicMemoryRecallCoordinator,
    MemoryRecallStatus,
    RecallBudget,
)
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextSource,
    CognitiveContextStatus,
    CognitiveContextBudget,
    CognitiveContextRequest,
    CognitiveContextItem,
    CognitiveContextTelemetry,
    CognitiveContextBundle,
    UnifiedCognitiveContextComposer,
)
from chakrview.cognition.repository.neural_adapter import (
    MockNeuralProposalAdapter,
    NeuralProposalOutput,
)


def _build_test_repo() -> Tuple[RepositoryState, RepositoryContextStore]:
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="test_cognitive_repo",
            project_name="Test Cognitive Repo",
            description="Testing unified cognitive context composition",
            entrypoint="main.py",
        ),
        files=[
            SourceFile(
                path="billing/tax.py",
                content="def compute_tax(subtotal: float, rate: float = 0.20) -> float:\n    return subtotal * rate\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/discount.py",
                content="def compute_discount(subtotal: float, vip: bool = False) -> float:\n    return 0.10 * subtotal if vip else 0.0\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/service.py",
                content="from billing.tax import compute_tax\nfrom billing.discount import compute_discount\n\ndef invoice(amt: float) -> float:\n    return compute_tax(amt) - compute_discount(amt)\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    context_store = RepositoryContextStore(project_id="test_cognitive_repo")
    context_store.synchronize(manifest)
    return repo_state, context_store


def _create_record(
    memory_id: str,
    task_family: str,
    repository_pattern: str,
    solution_pattern: str,
    affected_modules: List[str],
    successful_episodes: int = 1,
    failed_episodes: int = 0,
    known_boundaries: Optional[List[str]] = None,
    superseded_by: Optional[str] = None,
    confidence: float = 0.9,
) -> RepositorySemanticRecord:
    return RepositorySemanticRecord(
        memory_id=memory_id,
        task_family=task_family,
        language="python",
        framework="standard_library",
        repository_pattern=repository_pattern,
        symptom_signature=f"Symptom for {repository_pattern}",
        root_cause_signature=f"Root cause for {repository_pattern}",
        dependency_signature="billing -> models",
        affected_modules=affected_modules,
        solution_pattern=solution_pattern,
        verification_requirements=["pytest tests/"],
        known_boundaries=known_boundaries or [],
        confidence=confidence,
        evidence_count=successful_episodes + failed_episodes,
        successful_episodes=successful_episodes,
        failed_episodes=failed_episodes,
        superseded_by=superseded_by,
    )



def test_baseline_freeze_step67():
    """Verify neural baseline remains bit-exact."""
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    m.eval()
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    assert h.hexdigest() == EXPECTED_WEIGHT_HASH
    assert sum(p.numel() for p in m.parameters()) == 3_443_136


def test_repository_evidence_composition():
    """Verify current repository evidence is grounded and included in positive evidence."""
    repo_state, context_store = _build_test_repo()
    composer = UnifiedCognitiveContextComposer(context_store=context_store)

    req = CognitiveContextRequest(
        task_description="Refactor compute_tax to support tax exemptions in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
        target_symbols=("compute_tax",),
    )
    bundle = composer.compose_context(req)

    assert not bundle.abstained
    assert bundle.has_positive_context
    assert len(bundle.positive_evidence) > 0
    # Every evidence item must have source_type REPOSITORY_EVIDENCE
    for ev in bundle.positive_evidence:
        assert ev.source_type == CognitiveContextSource.REPOSITORY_EVIDENCE
        assert ev.status == CognitiveContextStatus.ACTIVE
        assert ev.epistemic_state == EpistemicState.KNOWN


def test_valid_episodic_memory_included():
    """Verify valid recalled episodic memory is composed into positive memories."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_ok",
            task_family="billing_refactor",
            repository_pattern="tax calculation exemption pattern",
            solution_pattern="def compute_tax(subtotal, rate=0.20, exempt=False):\n    return 0.0 if exempt else subtotal * rate",
            affected_modules=["billing/tax.py"],
            successful_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor compute_tax to support tax exemptions in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
        target_symbols=("compute_tax",),
    )
    bundle = composer.compose_context(req)

    assert len(bundle.positive_memories) == 1
    item = bundle.positive_memories[0]
    assert item.source_identifier == "mem_tax_ok"
    assert item.status == CognitiveContextStatus.ACTIVE
    assert item.source_type == CognitiveContextSource.EPISODIC_MEMORY


def test_negative_boundary_isolated():
    """Verify negative failure pattern is strictly quarantined to negative_boundaries channel."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="neg_tax_rate_clamp",
            task_family="billing_refactor",
            repository_pattern="negative rate clamping bug",
            solution_pattern="DO_NOT_APPLY: Clamp tax rate to 0.0 without validation",
            affected_modules=["billing/tax.py"],
            known_boundaries=["DO_NOT_APPLY negative tax clamping"],
            successful_episodes=0,
            failed_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax rate clamping in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)

    assert len(bundle.negative_boundaries) == 1
    neg = bundle.negative_boundaries[0]
    assert neg.source_identifier == "neg_tax_rate_clamp"
    assert neg.status == CognitiveContextStatus.NEGATIVE
    assert "DO_NOT_APPLY" in neg.content_payload
    # Must NOT appear in positive memories
    assert not any(p.source_identifier == "neg_tax_rate_clamp" for p in bundle.positive_memories)


def test_stale_memory_excluded():
    """Verify memory targeting deleted or drifted module is excluded from positive context."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_stale_old_auth",
            task_family="billing_refactor",
            repository_pattern="auth check for billing",
            solution_pattern="def check_auth(): pass",
            affected_modules=["billing/legacy_auth.py"],  # does not exist in repo
            successful_episodes=1,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Update auth in billing",
        task_family="billing_refactor",
    )
    bundle = composer.compose_context(req)

    assert len(bundle.positive_memories) == 0
    # Either in excluded_items or conflicted_items
    all_ex = bundle.excluded_items + bundle.conflicted_items
    assert any(x.source_identifier == "mem_stale_old_auth" for x in all_ex)


def test_superseded_memory_excluded():
    """Verify superseded memory is excluded from positive context."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_v1",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="def compute_tax_v1(): pass",
            affected_modules=["billing/tax.py"],
            superseded_by="mem_tax_v2",
            successful_episodes=1,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax calculation in billing/tax.py",
        task_family="billing_refactor",
    )
    bundle = composer.compose_context(req)

    assert not any(p.source_identifier == "mem_tax_v1" for p in bundle.positive_memories)
    assert any(x.source_identifier == "mem_tax_v1" and x.status == CognitiveContextStatus.SUPERSEDED for x in bundle.excluded_items)


def test_positive_collides_with_negative_boundary():
    """Verify positive memory colliding with a negative boundary on same module is quarantined as CONFLICTED."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="pos_tax_mod",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="def compute_tax(subtotal): return subtotal * 0.15",
            affected_modules=["billing/tax.py"],
            successful_episodes=3,
        )
    )
    mem_index.insert(
        _create_record(
            memory_id="neg_tax_mod",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="DO_NOT_APPLY: modification on billing/tax.py produces failure",
            affected_modules=["billing/tax.py"],
            known_boundaries=["DO_NOT_APPLY tax modification without approval"],
            successful_episodes=0,
            failed_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)

    assert not any(p.source_identifier == "pos_tax_mod" for p in bundle.positive_memories)
    assert any(c.source_identifier == "pos_tax_mod" and c.status == CognitiveContextStatus.CONFLICTED for c in bundle.conflicted_items)


def test_explicit_task_constraint_conflict():
    """Verify explicit task constraint forbidding a module quarantines positive memory as CONFLICTED."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="pos_discount_mod",
            task_family="discount_refactor",
            repository_pattern="discount calculation",
            solution_pattern="def compute_discount(): return 0.2",
            affected_modules=["billing/discount.py"],
            successful_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor discount without modifying discount file",
        task_family="discount_refactor",
        explicit_constraints=("FORBID_BILLING/DISCOUNT.PY",),
    )
    bundle = composer.compose_context(req)

    assert not any(p.source_identifier == "pos_discount_mod" for p in bundle.positive_memories)
    assert any(c.source_identifier == "pos_discount_mod" and c.status == CognitiveContextStatus.CONFLICTED for c in bundle.conflicted_items)


def test_repository_fingerprint_mismatch_abstention():
    """Verify requested fingerprint mismatch triggers deterministic abstention."""
    repo_state, context_store = _build_test_repo()
    composer = UnifiedCognitiveContextComposer(context_store=context_store)

    req = CognitiveContextRequest(
        task_description="Perform update",
        task_family="billing_refactor",
        repository_fingerprint="00000000000000000000000000000000",  # wrong fingerprint
    )
    bundle = composer.compose_context(req)

    assert bundle.abstained
    assert "Repository fingerprint mismatch" in (bundle.abstain_reason or "")
    assert not bundle.has_positive_context


def test_complete_provenance_invariant():
    """Verify every CognitiveContextItem possesses valid, non-empty provenance metadata."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_prov",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="def compute_tax(): pass",
            affected_modules=["billing/tax.py"],
            successful_episodes=1,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor compute_tax in billing/tax.py",
        task_family="billing_refactor",
    )
    bundle = composer.compose_context(req)

    all_items = (
        bundle.positive_evidence
        + bundle.positive_memories
        + bundle.negative_boundaries
        + bundle.conflicted_items
        + bundle.excluded_items
    )
    assert len(all_items) > 0
    for it in all_items:
        assert it.item_id != ""
        assert it.source_identifier != ""
        assert it.evidence_fingerprint != ""
        assert it.deterministic_reason != ""
        assert it.epistemic_state is not None
        assert it.ordering_key != ""


def test_budget_enforcement():
    """Verify budget limits are strictly enforced on evidence, memories, boundaries, and total items."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    for i in range(10):
        mem_index.insert(
            _create_record(
                memory_id=f"mem_budget_{i}",
                task_family="billing_refactor",
                repository_pattern=f"tax variation {i}",
                solution_pattern=f"# pattern {i}",
                affected_modules=["billing/tax.py"],
                successful_episodes=1,
            )
        )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    strict_budget = CognitiveContextBudget(
        max_evidence_items=2,
        max_positive_memories=2,
        max_negative_boundaries=1,
        max_total_items=4,
    )
    req = CognitiveContextRequest(
        task_description="Refactor billing tax",
        task_family="billing_refactor",
        budget=strict_budget,
    )
    bundle = composer.compose_context(req)

    assert len(bundle.positive_evidence) <= strict_budget.max_evidence_items
    assert len(bundle.positive_memories) <= strict_budget.max_positive_memories
    assert len(bundle.negative_boundaries) <= strict_budget.max_negative_boundaries
    assert bundle.total_active_items <= strict_budget.max_total_items


def test_deterministic_repeatability():
    """Verify identical inputs produce bit-for-bit identical CognitiveContextBundle."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_repeat",
            task_family="billing_refactor",
            repository_pattern="tax pattern",
            solution_pattern="def compute_tax(): pass",
            affected_modules=["billing/tax.py"],
            successful_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax billing",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    b1 = composer.compose_context(req)
    b2 = composer.compose_context(req)

    assert b1.telemetry.context_fingerprint == b2.telemetry.context_fingerprint
    assert len(b1.positive_evidence) == len(b2.positive_evidence)
    assert len(b1.positive_memories) == len(b2.positive_memories)
    assert [x.item_id for x in b1.positive_evidence] == [x.item_id for x in b2.positive_evidence]
    assert [x.item_id for x in b1.positive_memories] == [x.item_id for x in b2.positive_memories]


def test_neural_context_passive_data_only():
    """Verify to_neural_context emits strictly passive data dictionaries with no executable handles."""
    repo_state, context_store = _build_test_repo()
    composer = UnifiedCognitiveContextComposer(context_store=context_store)

    req = CognitiveContextRequest(
        task_description="Refactor tax billing",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)
    neural_ctx = bundle.to_neural_context()

    assert isinstance(neural_ctx, dict)
    assert "task_description" in neural_ctx
    assert "positive_evidence" in neural_ctx
    assert "positive_solution_patterns" in neural_ctx
    assert "negative_boundaries" in neural_ctx

    # Assert immutability / lack of callbacks
    for k, v in neural_ctx.items():
        assert not callable(v)
        assert not hasattr(v, "mutate")
        assert not hasattr(v, "execute")
        assert not hasattr(v, "write")


def test_neural_adapter_integration():
    """Verify MockNeuralProposalAdapter consumes to_neural_context and generates proposal."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_prop",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="def compute_tax(subtotal, rate=0.20):\n    return subtotal * rate * 0.9",
            affected_modules=["billing/tax.py"],
            successful_episodes=2,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax billing",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)
    neural_ctx = bundle.to_neural_context()

    adapter = MockNeuralProposalAdapter()
    proposal = adapter.generate_proposal(neural_ctx)

    assert proposal is not None
    assert proposal.objective == req.task_description
    assert proposal.target_files == ["billing/tax.py"]
    assert "billing/tax.py" in proposal.proposed_patches
    assert "0.9" in proposal.proposed_patches["billing/tax.py"]


def test_empty_memory_composition():
    """Verify composition functions cleanly when memory index is completely empty."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor compute_tax in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)

    assert not bundle.abstained
    assert len(bundle.positive_evidence) > 0
    assert len(bundle.positive_memories) == 0
    assert len(bundle.negative_boundaries) == 0
    assert len(bundle.conflicted_items) == 0


def test_neural_context_cannot_mutate_memory_index():
    """Verify passing neural context to adapter cannot mutate or append to RepositoryMemoryIndex."""
    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()
    mem_index.insert(
        _create_record(
            memory_id="mem_tax_immut",
            task_family="billing_refactor",
            repository_pattern="tax calculation",
            solution_pattern="def compute_tax(): pass",
            affected_modules=["billing/tax.py"],
            successful_episodes=1,
        )
    )
    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    req = CognitiveContextRequest(
        task_description="Refactor tax billing",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
    )
    bundle = composer.compose_context(req)
    neural_ctx = bundle.to_neural_context()

    adapter = MockNeuralProposalAdapter()
    _ = adapter.generate_proposal(neural_ctx)

    # RepositoryMemoryIndex must remain unchanged
    assert len(mem_index.candidate_set(task_family="billing_refactor")) == 1
    assert mem_index.lookup("mem_tax_immut") is not None
    assert mem_index.lookup("mem_tax_immut").successful_episodes == 1

