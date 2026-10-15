"""
Batches 356-363: Multi-Hop Evidence Chain and Reasoning Test Suite.
Location: tests/chakrverify/test_multihop_verification.py

Tests:
1. End-to-end 2-hop verified evidence chain (Doc A -> Bridge Entity -> Doc B -> Target Claim).
2. Broken evidence chain abstention (Hop 1 succeeds, Hop 2 missing -> INSUFFICIENT_EVIDENCE with broken_hop_index=1).
3. Hop 1 ungrounded abstention (start entity missing -> INSUFFICIENT_EVIDENCE with broken_hop_index=0).
4. Hop 2 contradiction detection (Hop 1 succeeds, Hop 2 asserts opposite relation -> CONTRADICTED).
5. Exact character offsets verification across both hops.
6. Multi-hop CLI execution with structured JSON output export.
"""
from pathlib import Path
import json
import pytest
import subprocess
import sys

sys.path.insert(0, ".")

from chakrverify.core import (
    MultiHopVerificationEngine,
    VerificationVerdict,
)


@pytest.fixture
def multihop_docs(tmp_path: Path):
    doc_a = tmp_path / "chip_design_spec.txt"
    doc_a.write_text(
        "Hardware Architecture Module A.\n"
        "The primary neural accelerator is designed with ArcticPro silicon.\n"
        "ArcticPro silicon is fabricated by Semiconductor Foundry Seven.\n",
        encoding="utf-8"
    )

    doc_b = tmp_path / "foundry_report.txt"
    doc_b.write_text(
        "Global Foundry Inspection Summary.\n"
        "Semiconductor Foundry Seven has verified high yield and operates in Dresden Germany.\n"
        "Semiconductor Foundry Eight suffered complete power grid shutdown.\n",
        encoding="utf-8"
    )

    doc_b_conflict = tmp_path / "foundry_denial.txt"
    doc_b_conflict.write_text(
        "Correction Bulletin.\n"
        "Semiconductor Foundry Seven does not operate in Dresden Germany; it was relocated to Singapore.\n",
        encoding="utf-8"
    )

    return {"doc_a": doc_a, "doc_b": doc_b, "doc_conflict": doc_b_conflict}


def test_multihop_verified_chain(multihop_docs):
    engine = MultiHopVerificationEngine()
    engine.add_document(multihop_docs["doc_a"])
    engine.add_document(multihop_docs["doc_b"])

    # Hop 1: ArcticPro -> fabricated by -> Semiconductor Foundry Seven
    # Hop 2: Semiconductor Foundry Seven -> operates in Dresden Germany
    res = engine.verify_multi_hop_claim(
        start_entity="ArcticPro",
        intermediate_relation="fabricated by",
        target_claim="operates in Dresden Germany",
        candidate_bridges=["Semiconductor Foundry Seven", "Semiconductor Foundry Eight"],
    )

    assert res.verdict == VerificationVerdict.SUPPORTED
    assert res.confidence >= 0.85
    assert len(res.hops) == 2
    assert res.broken_hop_index is None

    # Check hop 0
    assert res.hops[0].target_entity == "Semiconductor Foundry Seven"
    assert "chip_design_spec.txt" in res.hops[0].source_doc_id
    assert res.hops[0].is_supported is True

    # Check hop 1
    assert "foundry_report.txt" in res.hops[1].source_doc_id
    assert res.hops[1].is_supported is True

    # Check exact offsets against raw file text
    raw_a = multihop_docs["doc_a"].read_text(encoding="utf-8")
    hop0 = res.hops[0]
    assert raw_a[hop0.start_char:hop0.end_char] == hop0.passage

    raw_b = multihop_docs["doc_b"].read_text(encoding="utf-8")
    hop1 = res.hops[1]
    assert raw_b[hop1.start_char:hop1.end_char] == hop1.passage


def test_multihop_broken_chain_abstention(multihop_docs):
    engine = MultiHopVerificationEngine()
    engine.add_document(multihop_docs["doc_a"])
    # Do NOT add doc_b (Hop 2 is missing)

    res = engine.verify_multi_hop_claim(
        start_entity="ArcticPro",
        intermediate_relation="fabricated by",
        target_claim="operates in Dresden Germany",
        candidate_bridges=["Semiconductor Foundry Seven"],
    )

    assert res.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert res.broken_hop_index == 1
    assert "Broken evidence chain at Hop 2" in res.explanation
    assert len(res.hops) == 1
    assert res.hops[0].status == "SUPPORTED"


def test_multihop_unsupported_start_entity(multihop_docs):
    engine = MultiHopVerificationEngine()
    engine.add_document(multihop_docs["doc_a"])
    engine.add_document(multihop_docs["doc_b"])

    res = engine.verify_multi_hop_claim(
        start_entity="NonExistentAcceleratorX",
        intermediate_relation="fabricated by",
        target_claim="operates in Dresden Germany",
    )

    assert res.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert res.broken_hop_index == 0
    assert len(res.hops) == 0


def test_multihop_hop2_contradiction(multihop_docs):
    engine = MultiHopVerificationEngine()
    engine.add_document(multihop_docs["doc_a"])
    engine.add_document(multihop_docs["doc_conflict"])

    # Claim asserts Foundry Seven operates in Dresden, but doc_conflict asserts it does not
    res = engine.verify_multi_hop_claim(
        start_entity="ArcticPro",
        intermediate_relation="fabricated by",
        target_claim="operates in Dresden Germany",
        candidate_bridges=["Semiconductor Foundry Seven"],
    )

    assert res.verdict == VerificationVerdict.CONTRADICTED
    assert res.has_conflicts is True
    assert res.broken_hop_index == 1
    assert len(res.hops) == 2
    assert res.hops[1].status == "CONTRADICTED"


def test_multihop_cli_execution(multihop_docs, tmp_path: Path):
    out_json = tmp_path / "multihop_report.json"
    cmd = [
        sys.executable,
        "-m", "chakrverify.cli",
        "--docs", str(multihop_docs["doc_a"]), str(multihop_docs["doc_b"]),
        "--multihop",
        "--start-entity", "ArcticPro",
        "--relation", "fabricated by",
        "--target-claim", "operates in Dresden Germany",
        "--bridges", "Semiconductor Foundry Seven",
        "--output", str(out_json),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["verdict"] == "SUPPORTED"
    assert len(data[0]["hops"]) == 2
