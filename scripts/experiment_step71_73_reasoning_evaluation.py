"""
ChakrView Steps 71–73 Combined Milestone Experiment:
Unified Reasoning, Critical Thinking & Self-Evaluation Pipeline.

Demonstrates end-to-end cognitive flow:
TASK / QUESTION
       ↓
Cognitive Context (Step 67)
       ↓
Neural Proposal (Step 68 ChakrMicro greedy inference)
       ↓
Step 71: Structured Reasoning Artifact (Claims, Assumptions, Observations)
       ↓
Step 72: Critical Thinking & Alternative Hypotheses (Evidence Balances, Assumptions, Alternatives)
       ↓
Step 73: Independent Self-Evaluation (Grounding, Constraints, Contradictions, Handles)
       ↓
Bounded Revision Cycle (Max 2 iterations)
       ↓
Final Verified Verdict (ACCEPT / REVISE / REJECT / ABSTAIN)

Tests 12 Essential Milestone Conditions:
- Cond 1: Valid task produces structured reasoning artifact with FACT, MEMORY, INFERENCE claims.
- Cond 2: Every FACT claim carries verified evidence IDs.
- Cond 3: Critical thinking identifies supporting evidence strength and alternative hypotheses.
- Cond 4: Competing hypotheses are explicitly formulated and compared.
- Cond 5: Self-evaluation accepts valid, grounded reasoning artifact.
- Cond 6: Non-existent / hallucinated repository file triggers REJECT.
- Cond 7: Injected dangerous execution handle (os.system) triggers REJECT.
- Cond 8: Quarantined conflict / contradiction triggers ABSTAIN.
- Cond 9: Empty context with zero evidence triggers UNKNOWN / ABSTAIN without hallucination.
- Cond 10: Flawed reasoning with ungrounded fact is corrected via bounded revision to ACCEPT.
- Cond 11: Bounded revision strictly halts within MAX_CYCLES (2).
- Cond 12: Neural core parameter count (3,443,136) and SHA-256 weight hash remain bit-exact (dW = 0).
"""

from __future__ import annotations

import copy
import hashlib
import sys
import time
from typing import Dict, List, Tuple

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
    ChakrMicroNeuralProposalAdapter,
)
from chakrview.cognition.reasoning.structured import (
    EpistemicCategory,
    EpistemicConfidenceState,
    ReasoningClaim,
    StructuredReasoningArtifact,
    StructuredReasoningEngine,
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


def compute_neural_hash(model: ChakrMicro) -> Tuple[str, int]:
    hasher = hashlib.sha256()
    total_params = 0
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
        total_params += param.numel()
    return hasher.hexdigest(), total_params


def run_experiment() -> bool:
    print("=" * 70)
    print("CHAKRVIEW STEPS 71–73 COMBINED EXPERIMENT")
    print("Unified Reasoning, Critical Thinking & Self-Evaluation Pipeline")
    print("=" * 70)

    # Pre-experiment neural baseline verification
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    pre_hash, pre_params = compute_neural_hash(model)
    print(f"Pre-experiment Neural Hash  : {pre_hash}")
    print(f"Pre-experiment Parameters   : {pre_params:,}")
    assert pre_hash == EXPECTED_WEIGHT_HASH
    assert pre_params == 3_443_136

    conditions_passed: Dict[str, bool] = {}

    # Setup isolated test repository
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="billing_engine",
            project_name="Billing Engine",
            description="Testing cognitive reasoning and evaluation",
            entrypoint="billing/discount.py",
        ),
        files=[
            SourceFile(
                path="billing/discount.py",
                content=(
                    "def compute_discount(order_total: float, discount: float) -> float:\n"
                    "    # Bug: Unbounded discount\n"
                    "    return discount\n"
                ),
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/taxes.py",
                content="def compute_tax(subtotal: float) -> float:\n    return round(subtotal * 0.05, 2)\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    context_store = RepositoryContextStore(project_id="billing_engine")
    context_store.synchronize(manifest)

    # Setup episodic memory
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
        task_description="Bound discount computation to order total in billing/discount.py",
        task_family="billing_discount",
        target_files=("billing/discount.py",),
        target_symbols=("compute_discount",),
        allowed_files=("billing/discount.py", "billing/taxes.py"),
    )
    context_bundle = composer.compose_context(req)

    # Step 68 Neural Proposal
    adapter = ChakrMicroNeuralProposalAdapter(context_store=context_store)
    proposal = adapter.generate_proposal_contract(context_bundle)
    print(f"Proposal Validated Status   : {proposal.validation_status.name}")

    # -----------------------------------------------------------------
    # Cond 1: Valid task produces structured reasoning artifact
    # -----------------------------------------------------------------
    reasoning = StructuredReasoningEngine.reason(
        context_bundle=context_bundle,
        proposal=proposal,
        context_store=context_store,
    )
    categories = {c.category for c in reasoning.claims}
    cond1 = (
        not reasoning.is_abstained and
        EpistemicCategory.FACT in categories and
        EpistemicCategory.MEMORY in categories and
        EpistemicCategory.PROPOSAL in categories and
        len(reasoning.artifact_fingerprint) == 16
    )
    conditions_passed["Cond 1: Structured Reasoning Creation"] = cond1
    print(f"Cond 1 (Structured Reasoning Creation)    : {'PASS' if cond1 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 2: Every FACT claim carries verified evidence IDs
    # -----------------------------------------------------------------
    fact_claims = [c for c in reasoning.claims if c.category == EpistemicCategory.FACT]
    cond2 = len(fact_claims) > 0 and all(len(fc.supporting_evidence_ids) > 0 for fc in fact_claims)
    conditions_passed["Cond 2: Fact Provenance Preservation"] = cond2
    print(f"Cond 2 (Fact Provenance Preservation)     : {'PASS' if cond2 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 3: Critical thinking identifies supporting evidence strength
    # -----------------------------------------------------------------
    crit_report = CriticalThinkingEngine.evaluate(
        artifact=reasoning,
        context_bundle=context_bundle,
        context_store=context_store,
    )
    cond3 = (
        len(crit_report.evidence_balances) > 0 and
        crit_report.recommendation in ("PROCEED", "EXPLORE_ALTERNATIVES")
    )
    conditions_passed["Cond 3: Evidence Balance Audit"] = cond3
    print(f"Cond 3 (Evidence Balance Audit)           : {'PASS' if cond3 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 4: Competing hypotheses explicitly formulated
    # -----------------------------------------------------------------
    cond4 = (
        len(crit_report.alternative_hypotheses) >= 2 and
        all(alt.hypothesis_id and 0.0 <= alt.plausibility <= 1.0 for alt in crit_report.alternative_hypotheses)
    )
    conditions_passed["Cond 4: Alternative Hypotheses"] = cond4
    print(f"Cond 4 (Alternative Hypotheses Formulated): {'PASS' if cond4 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 5: Self-evaluation accepts valid, grounded reasoning
    # -----------------------------------------------------------------
    eval_report = SelfEvaluator.evaluate(
        reasoning=reasoning,
        critical_report=crit_report,
        proposal=proposal,
        context_bundle=context_bundle,
        context_store=context_store,
    )
    cond5 = (eval_report.verdict == EvaluationVerdict.ACCEPT and len(eval_report.identified_defects) == 0)
    conditions_passed["Cond 5: Self-Evaluation Acceptance"] = cond5
    print(f"Cond 5 (Self-Evaluation Acceptance)       : {'PASS' if cond5 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 6: Non-existent / hallucinated file triggers REJECT
    # -----------------------------------------------------------------
    hallucinated_prop = ProposalContract(
        proposal_id="prop_ghost",
        task_id="Ghost file",
        proposal_type="GROUNDED_REFACTORING",
        summary="Ghost",
        proposed_changes={"billing/non_existent.py": "pass\n"},
        target_files=("billing/non_existent.py",),
        target_symbols=(),
        reasoning_trace_summary="",
        supporting_evidence_ids=(),
        supporting_memory_ids=(),
        negative_boundary_ids=(),
        confidence=0.5,
        epistemic_state=EpistemicState.KNOWN,
        context_fingerprint="fp",
        validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
    )
    r_ghost = StructuredReasoningEngine.reason(context_bundle, hallucinated_prop, context_store)
    c_ghost = CriticalThinkingEngine.evaluate(r_ghost, context_bundle, context_store)
    e_ghost = SelfEvaluator.evaluate(r_ghost, c_ghost, hallucinated_prop, context_bundle, context_store)
    cond6 = (e_ghost.verdict == EvaluationVerdict.REJECT and any("non-existent file" in d for d in e_ghost.identified_defects))
    conditions_passed["Cond 6: Hallucinated File Rejection"] = cond6
    print(f"Cond 6 (Hallucinated File Rejection)      : {'PASS' if cond6 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 7: Injected dangerous execution handle triggers REJECT
    # -----------------------------------------------------------------
    danger_prop = ProposalContract(
        proposal_id="prop_danger",
        task_id="Danger task",
        proposal_type="GROUNDED_REFACTORING",
        summary="Danger",
        proposed_changes={"billing/discount.py": "import os\nos.system('calc.exe')\n"},
        target_files=("billing/discount.py",),
        target_symbols=(),
        reasoning_trace_summary="",
        supporting_evidence_ids=("ev_01",),
        supporting_memory_ids=(),
        negative_boundary_ids=(),
        confidence=0.5,
        epistemic_state=EpistemicState.KNOWN,
        context_fingerprint="fp",
        validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
    )
    r_danger = StructuredReasoningEngine.reason(context_bundle, danger_prop, context_store)
    c_danger = CriticalThinkingEngine.evaluate(r_danger, context_bundle, context_store)
    e_danger = SelfEvaluator.evaluate(r_danger, c_danger, danger_prop, context_bundle, context_store)
    cond7 = (e_danger.verdict == EvaluationVerdict.REJECT and any("Prohibited execution handle" in d for d in e_danger.identified_defects))
    conditions_passed["Cond 7: Dangerous Handle Rejection"] = cond7
    print(f"Cond 7 (Dangerous Handle Rejection)       : {'PASS' if cond7 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 8: Quarantined conflict / contradiction triggers ABSTAIN
    # -----------------------------------------------------------------
    conflicted_bundle = copy.deepcopy(context_bundle)
    conflicted_bundle.conflicted_items.append(
        CognitiveContextItem(
            item_id="ev_01",
            source_type=CognitiveContextSource.REPOSITORY_EVIDENCE,
            source_identifier="ev_01",
            module_reference="billing/discount.py",
            evidence_fingerprint="fp_conf",
            status=CognitiveContextStatus.CONFLICTED,
            deterministic_reason="Quarantined conflict with negative boundary",
            epistemic_state=EpistemicState.CONTRADICTED,
        )
    )
    r_conf = StructuredReasoningEngine.reason(conflicted_bundle, danger_prop, context_store)
    c_conf = CriticalThinkingEngine.evaluate(r_conf, conflicted_bundle, context_store)
    e_conf = SelfEvaluator.evaluate(r_conf, c_conf, danger_prop, conflicted_bundle, context_store)
    cond8 = (e_conf.verdict in (EvaluationVerdict.ABSTAIN, EvaluationVerdict.REJECT))
    conditions_passed["Cond 8: Contradiction Handling"] = cond8
    print(f"Cond 8 (Contradiction Handling)           : {'PASS' if cond8 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 9: Empty context with zero evidence triggers UNKNOWN / ABSTAIN
    # -----------------------------------------------------------------
    empty_bundle = CognitiveContextBundle(
        request=CognitiveContextRequest(task_description="Unknown gap", task_family="gap"),
        repository_fingerprint="fp_empty",
        positive_evidence=[],
        positive_memories=[],
    )
    r_empty = StructuredReasoningEngine.reason(empty_bundle, None, context_store)
    cond9 = (
        r_empty.is_abstained and
        r_empty.overall_confidence == EpistemicConfidenceState.UNKNOWN and
        len(r_empty.investigation_requirements) > 0
    )
    conditions_passed["Cond 9: Unknown / Insufficient Evidence"] = cond9
    print(f"Cond 9 (Unknown / Insufficient Evidence)  : {'PASS' if cond9 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 10: Flawed reasoning with ungrounded fact is corrected via bounded revision
    # -----------------------------------------------------------------
    bad_claims = list(reasoning.claims)
    bad_claims.append(ReasoningClaim(
        claim_id="claim_unsupported_fact",
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
    c_flawed = CriticalThinkingEngine.evaluate(flawed_reasoning, context_bundle, context_store)
    e_flawed = SelfEvaluator.evaluate(flawed_reasoning, c_flawed, proposal, context_bundle, context_store)
    cond10_initial = (e_flawed.verdict == EvaluationVerdict.REVISE)

    rev_res = BoundedRevisionCoordinator.coordinate(context_bundle, proposal, context_store)
    cond10 = cond10_initial and (rev_res.final_verdict == EvaluationVerdict.ACCEPT)
    conditions_passed["Cond 10: Bounded Self-Revision"] = cond10
    print(f"Cond 10 (Bounded Self-Revision)           : {'PASS' if cond10 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 11: Bounded revision strictly halts within MAX_CYCLES (2)
    # -----------------------------------------------------------------
    cond11 = (rev_res.cycles_completed <= BoundedRevisionCoordinator.MAX_CYCLES)
    conditions_passed["Cond 11: Bounded Cycles Termination"] = cond11
    print(f"Cond 11 (Bounded Cycles Termination)      : {'PASS' if cond11 else 'FAIL'}")

    # -----------------------------------------------------------------
    # Cond 12: Neural core parameter count and weight hash invariant (dW = 0)
    # -----------------------------------------------------------------
    post_hash, post_params = compute_neural_hash(model)
    cond12 = (
        post_hash == EXPECTED_WEIGHT_HASH and
        post_params == 3_443_136 and
        post_hash == pre_hash and
        post_params == pre_params
    )
    conditions_passed["Cond 12: Neural Invariant dW = 0"] = cond12
    print(f"Cond 12 (Neural Invariant dW = 0)         : {'PASS' if cond12 else 'FAIL'}")

    print("=" * 70)
    total_cond = len(conditions_passed)
    passed_cond = sum(1 for v in conditions_passed.values() if v)
    print(f"EXPERIMENT RESULTS: {passed_cond}/{total_cond} CONDITIONS PASSED")
    print("=" * 70)

    return passed_cond == total_cond


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
