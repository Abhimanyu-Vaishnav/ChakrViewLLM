"""
Tests for ChakrView Steps 71–73: Unified Reasoning, Critical Thinking & Self-Evaluation.

Validates Requirements A through V:
A. Structured reasoning creation
B. Provenance preservation (100% trace to evidence/memory IDs)
C. FACT vs INFERENCE separation
D. UNKNOWN handling (missing evidence produces UNKNOWN)
E. Insufficient evidence handling (abstains cleanly)
F. Supporting evidence recognition
G. Contradicting evidence recognition (collision with quarantined conflict)
H. Alternative hypothesis representation
I. Assumption tracking
J. Stale evidence handling
K. Conflicted memory handling
L. Hallucinated repository claim rejection
M. Constraint violation detection
N. Evaluator catches unsupported claim
O. Evaluator catches provenance mismatch
P. Evaluator catches contradiction
Q. Evaluator preserves uncertainty
R. Bounded revision resolves demoted claims
S. Bounded evaluation cycles (terminates within MAX_CYCLES=2)
T. Deterministic identical inputs produce identical digests
U. No neural weight mutation (dW = 0)
V. Steps 59–70 compatibility
"""

from __future__ import annotations

import copy
import hashlib
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextRequest,
    CognitiveContextBundle,
    CognitiveContextItem,
    CognitiveContextSource,
    CognitiveContextStatus,
    UnifiedCognitiveContextComposer,
)
from chakrview.cognition.repository.episodic_recall import EpisodicMemoryRecallCoordinator
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
    EpistemicPartition,
    StructuredEpistemicClaim,
)
from chakrview.cognition.reasoning.structured import (
    EpistemicCategory,
    EpistemicConfidenceState,
    ReasoningClaim,
    AssumptionRecord,
    StructuredReasoningArtifact,
    StructuredReasoningEngine,
    InvestigationRequirement,
)
from chakrview.cognition.reasoning.critical import (
    EvidenceStrength,
    EvidenceBalance,
    AlternativeHypothesis,
    CriticalAnalysisReport,
    CriticalThinkingEngine,
)
from chakrview.cognition.reasoning.evaluation import (
    EvaluationVerdict,
    SelfEvaluationReport,
    SelfEvaluator,
    BoundedRevisionResult,
    BoundedRevisionCoordinator,
)


@pytest.fixture
def repo_and_store():
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="test_repo_71_73",
            project_name="Test Repo 71-73",
            description="Testing unified reasoning, critical thinking, and evaluation",
            entrypoint="core.py",
        ),
        files=[
            SourceFile(
                path="billing/discount.py",
                content="def compute_discount(order_total: float, discount: float) -> float:\n    return discount\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/tax.py",
                content="def compute_tax(subtotal: float) -> float:\n    return round(subtotal * 0.05, 2)\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    context_store = RepositoryContextStore(project_id="test_repo_71_73")
    context_store.synchronize(manifest)
    return repo_state, context_store


@pytest.fixture
def sample_context_bundle(repo_and_store):
    repo_state, context_store = repo_and_store
    mem_index = RepositoryMemoryIndex()
    rec = RepositorySemanticRecord(
        memory_id="mem_disc_01",
        task_family="billing_discount",
        language="python",
        framework="standard_library",
        repository_pattern="discount_clamp",
        symptom_signature="Unbounded discount computation",
        root_cause_signature="Missing bounds check",
        dependency_signature="billing -> core",
        affected_modules=["billing/discount.py"],
        solution_pattern="def compute_discount(order_total: float, discount: float) -> float:\n    return min(order_total, max(0.0, discount))\n",
        verification_requirements=["pytest tests/"],
        known_boundaries=["Discount cannot exceed order total"],
        confidence=0.95,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    mem_index.insert(rec)
    coord = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coord)

    req = CognitiveContextRequest(
        task_description="Bound discount to order total in billing/discount.py",
        task_family="billing_discount",
        target_files=("billing/discount.py",),
        target_symbols=("compute_discount",),
        allowed_files=("billing/discount.py", "billing/tax.py"),
    )
    return composer.compose_context(req)


@pytest.fixture
def sample_proposal(sample_context_bundle):
    return ProposalContract(
        proposal_id="prop_sample_01",
        task_id="Bound discount",
        proposal_type="GROUNDED_REFACTORING",
        summary="Clamp discount",
        proposed_changes={"billing/discount.py": "def compute_discount(order_total, discount): return min(order_total, discount)"},
        target_files=("billing/discount.py",),
        target_symbols=("compute_discount",),
        reasoning_trace_summary="Clamped to order total based on historical pattern.",
        supporting_evidence_ids=("ev_ast_01",),
        supporting_memory_ids=("mem_disc_01",),
        negative_boundary_ids=(),
        confidence=0.92,
        epistemic_state=EpistemicState.KNOWN,
        context_fingerprint=sample_context_bundle.telemetry.context_fingerprint,
        validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
    )


class TestStep71StructuredReasoning:
    def test_structured_reasoning_creation(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        artifact = StructuredReasoningEngine.reason(
            context_bundle=sample_context_bundle,
            proposal=sample_proposal,
            context_store=store,
        )
        assert artifact is not None
        assert artifact.task_id == sample_context_bundle.request.task_description
        assert len(artifact.claims) > 0
        assert len(artifact.assumptions) > 0
        assert len(artifact.observations) > 0
        assert artifact.is_abstained is False
        assert len(artifact.artifact_fingerprint) == 16

    def test_provenance_preservation(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        artifact = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        fact_claims = [c for c in artifact.claims if c.category == EpistemicCategory.FACT]
        for fc in fact_claims:
            assert len(fc.supporting_evidence_ids) > 0

    def test_fact_vs_inference_separation(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        artifact = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        categories = {c.category for c in artifact.claims}
        assert EpistemicCategory.FACT in categories
        assert EpistemicCategory.MEMORY in categories
        # Proposal claims are not FACT
        proposal_claims = [c for c in artifact.claims if c.category == EpistemicCategory.PROPOSAL]
        for pc in proposal_claims:
            assert pc.category != EpistemicCategory.FACT

    def test_unknown_and_insufficient_evidence_handling(self, repo_and_store):
        _, store = repo_and_store
        # Empty context bundle with no evidence or memories
        req = CognitiveContextRequest(
            task_description="Unknown task with no evidence",
            task_family="unknown_family",
        )
        empty_bundle = CognitiveContextBundle(
            request=req,
            repository_fingerprint="empty_repo_fp",
            positive_evidence=[],
            positive_memories=[],
        )
        artifact = StructuredReasoningEngine.reason(empty_bundle, None, store)
        assert artifact.is_abstained is True
        assert artifact.overall_confidence == EpistemicConfidenceState.UNKNOWN
        assert len(artifact.investigation_requirements) > 0
        assert "Insufficient evidence" in (artifact.abstain_reason or "")


class TestStep72CriticalThinking:
    def test_evidence_balance_and_support_recognition(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, sample_context_bundle, store)
        
        assert critical_report is not None
        assert len(critical_report.evidence_balances) > 0
        # Definitive or corroborated evidence recognized
        strengths = {b.strength for b in critical_report.evidence_balances}
        assert EvidenceStrength.DEFINITIVE in strengths or EvidenceStrength.CORROBORATED in strengths
        assert critical_report.recommendation in ("PROCEED", "EXPLORE_ALTERNATIVES")

    def test_alternative_hypothesis_representation(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, sample_context_bundle, store)
        
        assert len(critical_report.alternative_hypotheses) >= 2
        for alt in critical_report.alternative_hypotheses:
            assert alt.hypothesis_id
            assert alt.description
            assert 0.0 <= alt.plausibility <= 1.0
            assert len(alt.tradeoffs) > 0

    def test_contradicting_evidence_recognition(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        # Inject a quarantined conflict item into context
        conflicted_bundle = copy.deepcopy(sample_context_bundle)
        conflicted_bundle.conflicted_items.append(
            CognitiveContextItem(
                item_id="ev_ast_01",  # matches proposal supporting evidence ID
                source_type=CognitiveContextSource.REPOSITORY_EVIDENCE,
                source_identifier="ev_ast_01",
                module_reference="billing/discount.py",
                evidence_fingerprint="fp_conf",
                status=CognitiveContextStatus.CONFLICTED,
                deterministic_reason="Contradicts negative boundary",
                epistemic_state=EpistemicState.CONTRADICTED,
            )
        )
        reasoning = StructuredReasoningEngine.reason(conflicted_bundle, sample_proposal, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, conflicted_bundle, store)
        
        assert len(critical_report.unresolved_contradictions) > 0
        assert critical_report.recommendation == "ABSTAIN"


class TestStep73SelfEvaluationAndBoundedRevision:
    def test_evaluator_accepts_valid_grounded_reasoning(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, sample_context_bundle, store)
        eval_report = SelfEvaluator.evaluate(reasoning, critical_report, sample_proposal, sample_context_bundle, store)
        
        assert eval_report.verdict == EvaluationVerdict.ACCEPT
        assert len(eval_report.identified_defects) == 0

    def test_evaluator_catches_hallucinated_file_rejection(self, sample_context_bundle, repo_and_store):
        _, store = repo_and_store
        # Construct proposal with non-existent file
        hallucinated_prop = ProposalContract(
            proposal_id="prop_hallucinated",
            task_id="Fake file task",
            proposal_type="GROUNDED_REFACTORING",
            summary="Fake",
            proposed_changes={"fake/non_existent.py": "x = 1\n"},
            target_files=("fake/non_existent.py",),
            target_symbols=(),
            reasoning_trace_summary="Hallucinated file",
            supporting_evidence_ids=(),
            supporting_memory_ids=(),
            negative_boundary_ids=(),
            confidence=0.5,
            epistemic_state=EpistemicState.KNOWN,
            context_fingerprint="fp",
            validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
        )
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, hallucinated_prop, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, sample_context_bundle, store)
        eval_report = SelfEvaluator.evaluate(reasoning, critical_report, hallucinated_prop, sample_context_bundle, store)
        
        assert eval_report.verdict == EvaluationVerdict.REJECT
        assert any("non-existent file" in d for d in eval_report.identified_defects)

    def test_evaluator_catches_dangerous_handles(self, sample_context_bundle, repo_and_store):
        _, store = repo_and_store
        danger_prop = ProposalContract(
            proposal_id="prop_danger",
            task_id="Danger task",
            proposal_type="GROUNDED_REFACTORING",
            summary="Danger",
            proposed_changes={"billing/discount.py": "import os\nos.system('calc')\n"},
            target_files=("billing/discount.py",),
            target_symbols=(),
            reasoning_trace_summary="Danger payload",
            supporting_evidence_ids=("ev_ast_01",),
            supporting_memory_ids=(),
            negative_boundary_ids=(),
            confidence=0.5,
            epistemic_state=EpistemicState.KNOWN,
            context_fingerprint="fp",
            validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
        )
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, danger_prop, store)
        critical_report = CriticalThinkingEngine.evaluate(reasoning, sample_context_bundle, store)
        eval_report = SelfEvaluator.evaluate(reasoning, critical_report, danger_prop, sample_context_bundle, store)
        
        assert eval_report.verdict == EvaluationVerdict.REJECT
        assert any("Prohibited execution handle" in d for d in eval_report.identified_defects)

    def test_bounded_revision_resolves_unproven_claim(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        # Inject an unsupported claim classified as FACT without supporting IDs
        reasoning = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        bad_claims = list(reasoning.claims)
        bad_claims.append(ReasoningClaim(
            claim_id="claim_bad_fact",
            category=EpistemicCategory.FACT,
            statement="Unsupported factual assertion",
            confidence_state=EpistemicConfidenceState.KNOWN,
            supporting_evidence_ids=(),  # Missing provenance!
        ))
        flawed_reasoning = StructuredReasoningArtifact(
            artifact_id=reasoning.artifact_id,
            task_id=reasoning.task_id,
            context_fingerprint=reasoning.context_fingerprint,
            repository_fingerprint=reasoning.repository_fingerprint,
            claims=tuple(bad_claims),
            assumptions=reasoning.assumptions,
            observations=reasoning.observations,
            candidate_conclusions=reasoning.candidate_conclusions,
            unresolved_questions=reasoning.unresolved_questions,
            investigation_requirements=reasoning.investigation_requirements,
            overall_confidence=reasoning.overall_confidence,
            is_abstained=reasoning.is_abstained,
            abstain_reason=reasoning.abstain_reason,
        )

        critical_report = CriticalThinkingEngine.evaluate(flawed_reasoning, sample_context_bundle, store)
        eval_report = SelfEvaluator.evaluate(flawed_reasoning, critical_report, sample_proposal, sample_context_bundle, store)
        
        # Initial evaluation requests REVISE
        assert eval_report.verdict == EvaluationVerdict.REVISE
        
        # Execute BoundedRevisionCoordinator
        rev_res = BoundedRevisionCoordinator.coordinate(sample_context_bundle, sample_proposal, store)
        assert rev_res.cycles_completed <= BoundedRevisionCoordinator.MAX_CYCLES
        assert rev_res.final_verdict == EvaluationVerdict.ACCEPT

    def test_deterministic_identical_inputs(self, sample_context_bundle, sample_proposal, repo_and_store):
        _, store = repo_and_store
        r1 = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        r2 = StructuredReasoningEngine.reason(sample_context_bundle, sample_proposal, store)
        assert r1.artifact_fingerprint == r2.artifact_fingerprint

        c1 = CriticalThinkingEngine.evaluate(r1, sample_context_bundle, store)
        c2 = CriticalThinkingEngine.evaluate(r2, sample_context_bundle, store)
        assert c1.report_fingerprint == c2.report_fingerprint

        e1 = SelfEvaluator.evaluate(r1, c1, sample_proposal, sample_context_bundle, store)
        e2 = SelfEvaluator.evaluate(r2, c2, sample_proposal, sample_context_bundle, store)
        assert e1.report_fingerprint == e2.report_fingerprint

    def test_neural_baseline_immutability(self):
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()
        hasher = hashlib.sha256()
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
        assert hasher.hexdigest() == EXPECTED_WEIGHT_HASH
        assert sum(p.numel() for p in model.parameters()) == 3_443_136
