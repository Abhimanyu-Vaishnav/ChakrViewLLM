"""
Unit and Integration Test Suite for Batches 261–270:
Open-Ended Question Generalization, Randomized Evaluation, Neural Capability Auditing, and Verification.
Location: tests/test_batch261_270_generalization.py
"""
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType
from chakrview.runtime.answer_verifier import AnswerVerifier
from chakrview.runtime.research_workflow import (
    PermissionGatedResearchWorkflow,
    ResearchState,
)
from scripts.founder_manual_test import ChakrViewRuntimeSession

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")


def test_canonical_invariants_strictly_immutable():
    """Verify canonical model checkpoint SHA-256 and parameter count remain strictly immutable."""
    import hashlib
    hasher = hashlib.sha256()
    with open(CANONICAL_CHECKPOINT, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    ck = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    m = ChakrMicro(ck["config"])
    m.load_state_dict(ck["model_state_dict"])
    assert sum(p.numel() for p in m.parameters()) == 3_443_136


def test_verbal_arithmetic_routing():
    """Verify verbal arithmetic phrases route accurately to CalculatorTool."""
    router = CognitiveRouter()

    # Subtraction from
    r_sub = router.classify_route("Subtract 36 from 78.")
    assert r_sub.route == RouteType.CALCULATOR_TOOL
    assert r_sub.metadata.get("expression") == "78 - 36"

    # Verbal sum of
    r_sum = router.classify_route("Compute sum of 62 and 14.")
    assert r_sum.route == RouteType.CALCULATOR_TOOL
    assert r_sum.metadata.get("expression") == "62 + 14"

    # Verbal multiply by
    r_mul = router.classify_route("Multiply 12 by 5.")
    assert r_mul.route == RouteType.CALCULATOR_TOOL
    assert r_mul.metadata.get("expression") == "12 * 5"


def test_generalized_celestial_false_premise():
    """Verify false premise challenges for leadership / government of celestial bodies."""
    router = CognitiveRouter()
    r_pm_mars = router.classify_route("Name the prime minister of Mars.")
    assert r_pm_mars.route == RouteType.CONCEPT_DEFINITION

    r_king_jup = router.classify_route("Who is the king of Jupiter?")
    assert r_king_jup.route == RouteType.CONCEPT_DEFINITION


def test_answer_verifier_catches_flaws():
    """Verify AnswerVerifier catches false premises, bad math, and unphysical temperatures."""
    # 1. Unchallenged false premise
    res_prem = AnswerVerifier.verify(
        query="What is the capital of Mars?",
        candidate_answer="The capital city of Mars is Utopia Planitia.",
    )
    assert not res_prem.is_valid
    assert res_prem.epistemic_status == "CHALLENGED_FALSE_PREMISE"

    # 2. Impossible physical temperature
    res_phys = AnswerVerifier.verify(
        query="What is the temperature of the sample?",
        candidate_answer="The sensor reported -50 K.",
    )
    assert not res_phys.is_valid
    assert res_phys.epistemic_status == "PHYSICAL_INVARIANT_VIOLATION"

    # 3. Valid answer passes
    res_valid = AnswerVerifier.verify(
        query="What is 25 + 37?",
        candidate_answer="62",
        tool_output="62",
    )
    assert res_valid.is_valid
    assert res_valid.epistemic_status == "VERIFIED"


def test_permission_gated_research_workflow():
    """Verify research state machine enforces permission gate, offline safety, and provenance."""
    workflow = PermissionGatedResearchWorkflow(
        local_knowledge_store={"internal_server": "192.168.1.100"}
    )

    # 1. Local knowledge served directly without requesting network permission
    res_local = workflow.execute_research_flow("sess_1", "What is the internal_server address?")
    assert res_local.state == ResearchState.LOCAL_FIRST
    assert res_local.answer == "192.168.1.100"

    # 2. External query halts at PERMISSION_REQUIRED when no grant callback provided
    res_req = workflow.execute_research_flow("sess_1", "What are the latest discoveries from JWST?")
    assert res_req.state == ResearchState.PERMISSION_REQUIRED
    assert res_req.requires_user_action is True

    # 3. User denies permission
    res_denied = workflow.execute_research_flow(
        "sess_1",
        "What are the latest discoveries from JWST?",
        user_grant_callback=lambda: False,
    )
    assert res_denied.state == ResearchState.RESEARCH_DENIED
    assert "not authorized" in res_denied.answer.lower()

    # 4. User grants permission, but network is offline
    res_offline = workflow.execute_research_flow(
        "sess_2",
        "What are the latest discoveries from JWST?",
        user_grant_callback=lambda: True,
        network_available=False,
    )
    assert res_offline.state == ResearchState.OFFLINE_COMPLETED
    assert "offline" in res_offline.answer.lower()

    # 5. User grants permission and network is online -> completed with provenance
    res_online = workflow.execute_research_flow(
        "sess_3",
        "What are the latest discoveries from JWST?",
        user_grant_callback=lambda: True,
        network_available=True,
    )
    assert res_online.state == ResearchState.RESEARCH_COMPLETED
    assert len(res_online.sources) == 1
    assert res_online.sources[0].url.startswith("https://")
