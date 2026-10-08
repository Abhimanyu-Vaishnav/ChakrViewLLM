"""
Test suite for Waves 546-555: Open-World Knowledge, Multi-Source Verification,
Temporal Freshness, Tool Orchestration, and Honest Abstention.
"""

import pytest
import time
from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
import torch

from chakrview.cognition.open_world.knowledge_interface import (
    OpenWorldKnowledgeInterface,
    KnowledgeRequest,
    EvidenceItem,
    SourceType,
    VerificationStatus,
)
from chakrview.cognition.open_world.multi_source_verifier import MultiSourceVerifier
from chakrview.cognition.open_world.contradictory_store import (
    ContradictoryKnowledgeStore,
    ContradictionResolutionVerdict,
)
from chakrview.cognition.open_world.temporal_knowledge import (
    TemporalKnowledgeManager,
    TemporalFactRecord,
    TemporalValidity,
)
from chakrview.cognition.open_world.unknown_handler import (
    UnknownHandlingEngine,
    InquiryClass,
    CognitiveDecision,
)
from chakrview.cognition.open_world.tool_reasoning import (
    CognitiveToolOrchestrator,
    ToolType,
    ToolNecessity,
)
from chakrview.cognition.open_world.open_world_continual import OpenWorldContinualEngine


def test_canonical_baseline_hash_specification():
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["total_params"] == 3443136
    assert res["delta_w"] == 0


def test_wave546_547_attribution_and_provenance():
    interface = OpenWorldKnowledgeInterface()
    req = KnowledgeRequest(
        target_entity="Saturn",
        target_relation="ring system composition",
        query_text="What are Saturn's rings composed of?",
    )
    raw = [
        {
            "entity": "Saturn",
            "relation": "ring system composition",
            "value": "99% water ice",
            "text": "Saturn's rings are composed predominantly of water ice particles.",
            "reputation": 0.95,
        },
        # Entity mismatch
        {
            "entity": "Jupiter",
            "relation": "ring system composition",
            "value": "dust particles",
            "text": "Jupiter's faint rings are composed of small dust particles.",
            "reputation": 0.90,
        },
    ]
    cand = interface.process_external_evidence(req, raw)
    assert cand.provenance_chain_valid is True
    assert cand.attribution_valid is False
    assert cand.verification_status == VerificationStatus.REJECTED


def test_wave548_multi_source_syndication_detection():
    interface = OpenWorldKnowledgeInterface()
    req = KnowledgeRequest(target_entity="Alpha", target_relation="metric")
    raw = [
        {
            "source_id": "wire_1",
            "publisher": "Syndicate",
            "entity": "Alpha",
            "relation": "metric",
            "value": "100",
            "text": "Alpha is 100.",
        },
        {
            "source_id": "wire_2",
            "publisher": "Syndicate",
            "entity": "Alpha",
            "relation": "metric",
            "value": "100",
            "text": "Alpha is 100.",
        },
        {
            "source_id": "wire_3",
            "publisher": "Syndicate",
            "entity": "Alpha",
            "relation": "metric",
            "value": "100",
            "text": "Alpha is 100.",
        },
    ]
    cand = interface.process_external_evidence(req, raw)
    indep = MultiSourceVerifier.assess_independence(cand.evidence_items, interface.registered_sources)
    assert indep.syndication_or_copy_detected is True
    assert indep.is_genuinely_independent is False


def test_wave549_contradiction_persistence_and_resolution():
    store = ContradictoryKnowledgeStore()
    ev = EvidenceItem("ev1", "r1", "s1", "q", "Sample text", "hash", "Ent", "rel", "ValA", time.time())
    rec = store.record_contradiction("Ent", "rel", "ValA", [ev], "ValB", [ev])
    assert rec.status == ContradictionResolutionVerdict.UNRESOLVED_CONTRADICTION

    # Authoritative resolution
    authoritative_ev = [ev]
    authoritative_ev[0].claimed_value = "ValA"
    res = store.attempt_resolution(rec.record_id, authoritative_ev, arbitration_source_reputation=0.95)
    assert res.status == ContradictionResolutionVerdict.RESOLVED_SIDE_A


def test_wave550_temporal_freshness():
    mgr = TemporalKnowledgeManager()
    t = time.time()
    # Expired historical fact
    mgr.register_fact(TemporalFactRecord(
        fact_id="f1", entity="Country X", attribute="treaty status", value="active",
        is_time_varying=True, valid_until=t - 100, observed_at=t - 200
    ))
    v, _, _ = mgr.evaluate_freshness("Country X", "treaty status", query_timestamp=t)
    assert v == TemporalValidity.HISTORICAL


def test_wave551_unknown_handling_and_abstention():
    # Abstain on insufficient evidence
    res_f = UnknownHandlingEngine.resolve_inquiry("unknown query", InquiryClass.F_INSUFFICIENT_EVIDENCE)
    assert res_f.decision == CognitiveDecision.ABSTAIN
    assert res_f.is_abstention is True

    # Report conflict on contradictory
    res_e = UnknownHandlingEngine.resolve_inquiry("conflict query", InquiryClass.E_CONTRADICTORY)
    assert res_e.decision == CognitiveDecision.REPORT_CONFLICT


def test_wave552_tool_orchestration_and_recovery():
    orch = CognitiveToolOrchestrator()
    orch.register_tool(ToolType.CALCULATOR, lambda q: 10 + 20)

    # Success
    res = orch.execute_and_reason("Calculate 10 + 20")
    assert res.tool_result.success is True
    assert res.tool_result.raw_output == 30

    # Failure recovery
    res_fail = orch.execute_and_reason("Calculate 10 + 20", simulate_tool_failure=True)
    assert res_fail.recovery_strategy_used == "FAILOVER_QUALIFY"


def test_wave554_open_world_continual_learning():
    engine = OpenWorldContinualEngine()
    rep = engine.execute_continual_learning_cycle(
        cycle_id="c_test",
        entity="Mars Rover Curiosity",
        relation="landing location",
        external_raw_evidence=[
            {"source_id": "nasa", "reputation": 0.95, "value": "Gale Crater", "text": "Curiosity landed in Gale Crater."},
            {"source_id": "jpl", "reputation": 0.95, "value": "Gale Crater", "text": "Curiosity touched down in Gale Crater."},
        ]
    )
    assert rep.committed_to_active is True
    assert rep.model_delta_w == 0
    active_entry = engine.active_memory.get("mars rover curiosity:landing location")
    assert active_entry is not None
    assert active_entry.value == "Gale Crater"
