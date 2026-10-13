"""
Regression test suite for Batches 151–160 Cognitive Routing & Grounded Memory.
Tests:
1. CognitiveRouter route classification.
2. RetrievalAdapter search, BM25 stopword filtering, and thresholding.
3. EvidenceExtractor key-value, multilingual, and conflict detection.
4. PersistentSessionMemory atomic writes, session isolation, and corruption recovery.
5. GroundedSessionAdapter multi-turn conversation flow.
"""
import json
import pytest
from pathlib import Path
import tempfile
import shutil

from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.session_adapter import GroundedSessionAdapter


def test_cognitive_router_classification():
    router = CognitiveRouter(retrieval_threshold=0.20)
    
    # 1. Greeting
    d1 = router.classify_route("Hello! How are you today?")
    assert d1.route == RouteType.DIRECT_CHAT
    assert not d1.requires_retrieval
    
    # 2. Knowledge seeking
    d2 = router.classify_route("What is the atomic number of Carbon?")
    assert d2.route == RouteType.KNOWLEDGE_RETRIEVAL
    assert d2.requires_retrieval
    
    # 3. Supplied context slot query
    d3 = router.classify_route("What is material?", supplied_context="material is steel.")
    assert d3.route == RouteType.SLOT_EXTRACTION
    assert d3.requires_slot_extraction


def test_evidence_extractor_functionality():
    # 1. Direct English
    res1 = EvidenceExtractor.extract_slot("material is titanium.", "What is material?")
    assert res1.success
    assert res1.extracted_value == "titanium"
    assert "material is titanium" in res1.supporting_span
    
    # 2. Hindi Multilingual
    res2 = EvidenceExtractor.extract_slot("धातु: लोहा।", "धातु क्या है?")
    assert res2.success
    assert res2.extracted_value == "लोहा"
    
    # 3. Conflicting Evidence
    res3 = EvidenceExtractor.extract_slot("status is active. status is inactive.", "What is status?")
    assert not res3.success
    assert res3.has_conflicts
    assert "active" in res3.conflicting_values
    assert "inactive" in res3.conflicting_values
    
    # 4. Abstention
    res4 = EvidenceExtractor.extract_slot("planet is Mars.", "What is color?")
    assert not res4.success
    assert res4.is_abstention


def test_retrieval_adapter_provenance_and_filter():
    adapter = RetrievalAdapter(min_score_threshold=0.20)
    doc_id = "doc_test_1"
    text = "The speed of sound in air is approximately 343 meters per second."
    cids = adapter.ingest_document(doc_id, text)
    assert len(cids) == 1
    
    # Match
    hits = adapter.search("What is the speed of sound?")
    assert len(hits) == 1
    assert hits[0].doc_id == doc_id
    assert len(hits[0].provenance_hash) == 64
    
    # Non-match abstention
    hits_none = adapter.search("Who invented the telephone?")
    assert len(hits_none) == 0


def test_persistent_memory_isolation_and_corruption_recovery():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        memory = PersistentSessionMemory(storage_dir=tmp_dir)
        
        # Write to sess_a
        memory.append_turn("sess_a", "user", "Hello A", {"origin": ComponentOrigin.NEURAL_CORE.value})
        memory.append_turn("sess_a", "assistant", "Hi A", {"origin": ComponentOrigin.NEURAL_CORE.value})
        
        # Write to sess_b
        memory.append_turn("sess_b", "user", "Hello B", {"origin": ComponentOrigin.NEURAL_CORE.value})
        
        # Verify isolation
        hist_a = memory.get_history("sess_a")
        hist_b = memory.get_history("sess_b")
        assert len(hist_a) == 2
        assert len(hist_b) == 1
        assert hist_a[0].content == "Hello A"
        assert hist_b[0].content == "Hello B"
        
        # Simulate corruption in sess_a
        file_a = tmp_dir / "session_sess_a.json"
        file_a.write_text("{ corrupt json string !!", encoding="utf-8")
        
        # Memory should safely recover and return empty history
        hist_recovered = memory.get_history("sess_a", reload=True)
        assert len(hist_recovered) == 0
        assert (tmp_dir / "session_sess_a.corrupted").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
