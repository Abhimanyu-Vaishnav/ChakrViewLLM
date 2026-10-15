"""
Batches 336-343: ChakrVerify Local Alpha Verification Test Suite.
Location: tests/chakrverify/test_verification.py

Tests:
1. Direct support verification with exact character offset citation.
2. Contradiction detection (e.g. claim asserts planet has municipal capital, doc asserts uninhabited).
3. Insufficient evidence honest abstention (unknown entities).
4. Multi-document conflict detection (conflicting claims across 2 documents).
5. CLI execution and JSON report export.
"""
from pathlib import Path
import json
import pytest
import subprocess
import sys
import tempfile

sys.path.insert(0, ".")

from chakrverify.core import (
    ClaimVerificationEngine,
    DocumentIngestionEngine,
    VerificationVerdict,
)


@pytest.fixture
def sample_docs(tmp_path: Path):
    doc1 = tmp_path / "procurement_spec.txt"
    doc1.write_text(
        "Section 1. Hardware Architecture.\n"
        "The digital signal processor is built using a Field Programmable Gate Array (FPGA).\n"
        "Operating clock frequency is strictly verified at 450 MHz.\n"
        "Power consumption must not exceed 12 Watts under peak load.\n",
        encoding="utf-8"
    )

    doc2 = tmp_path / "astronomy_report.txt"
    doc2.write_text(
        "Solar System Survey: Outer Planets.\n"
        "Neptune is an ice giant planet with dynamic methane atmospheres.\n"
        "Mars is an uninhabited terrestrial planet without countries, governments, or human municipal capitals.\n",
        encoding="utf-8"
    )

    doc_conflict = tmp_path / "supplier_conflict.txt"
    doc_conflict.write_text(
        "Supplier Addendum.\n"
        "Operating clock frequency is not verified at 450 MHz; it failed timing closure.\n",
        encoding="utf-8"
    )

    return {"doc1": doc1, "doc2": doc2, "conflict": doc_conflict}


def test_document_ingestion_offsets(sample_docs):
    chunks = DocumentIngestionEngine.ingest_file(sample_docs["doc1"])
    assert len(chunks) > 0
    # Check that character offsets correspond to exact original text
    c0 = chunks[0]
    raw_text = sample_docs["doc1"].read_text(encoding="utf-8")
    assert raw_text[c0.start_char:c0.end_char] == c0.text


def test_supported_claim_with_citation(sample_docs):
    engine = ClaimVerificationEngine()
    engine.add_document(sample_docs["doc1"])

    res = engine.verify_claim("The digital signal processor is built using an FPGA")
    assert res.verdict == VerificationVerdict.SUPPORTED
    assert res.confidence >= 0.80
    assert len(res.citations) > 0
    assert "procurement_spec.txt" in res.citations[0].doc_id


def test_contradicted_claim(sample_docs):
    engine = ClaimVerificationEngine()
    engine.add_document(sample_docs["doc2"])

    # Claim asserts Mars has a capital city
    res = engine.verify_claim("Mars has a human municipal capital city")
    assert res.verdict == VerificationVerdict.CONTRADICTED
    assert len(res.citations) > 0
    assert "uninhabited" in res.citations[0].exact_text.lower()


def test_insufficient_evidence_abstention(sample_docs):
    engine = ClaimVerificationEngine()
    engine.add_document(sample_docs["doc1"])

    # Query about completely unrelated entity
    res = engine.verify_claim("The quantum computer operates at 15 millikelvin")
    assert res.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE
    assert len(res.citations) == 0


def test_multi_document_conflict(sample_docs):
    engine = ClaimVerificationEngine()
    engine.add_document(sample_docs["doc1"])
    engine.add_document(sample_docs["conflict"])

    res = engine.verify_claim("Operating clock frequency is verified at 450 MHz")
    assert res.verdict == VerificationVerdict.CONTRADICTED
    assert res.has_conflicts or len(res.citations) > 0


def test_cli_execution(sample_docs, tmp_path: Path):
    out_json = tmp_path / "cli_report.json"
    cmd = [
        sys.executable,
        "-m", "chakrverify.cli",
        "--docs", str(sample_docs["doc1"]),
        "--claim", "Power consumption must not exceed 12 Watts",
        "--output", str(out_json)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["verdict"] == "SUPPORTED"
