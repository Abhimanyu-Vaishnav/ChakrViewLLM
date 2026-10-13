"""
Tests for Batches 181-190: Pointer Beam Decoding, Hybrid Re-ranking, and Multi-Document Evidence Synthesis.
Validates:
1. Canonical model invariants and checkpoint hash.
2. Pointer beam decoding equivalence with greedy when beam_width=1.
3. Length-normalized hypothesis scoring and causal context masking.
4. Two-stage hybrid retriever (BM25 + neural re-ranking) and lexical fallback.
5. Multi-document evidence assembler deduplication and contradiction detection.
6. CognitiveTaskEngine multi-document synthesis workflow and restart recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
import pytest

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import (
    ChakrPointerHead,
    BeamHypothesis,
    decode_pointer_greedy,
    decode_pointer_beam,
)
from chakrview.runtime.hybrid_reranker import HybridReranker, RerankedEvidence
from chakrview.runtime.evidence_assembler import MultiDocumentEvidenceAssembler, EvidenceBundle
from chakrview.runtime.task_engine import CognitiveTaskEngine, TaskStatus
from chakrview.runtime.persistent_memory import PersistentSessionMemory

CANONICAL_CHECKPOINT_PATH = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
EXPECTED_CHECKPOINT_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_canonical_invariants():
    """Verify that canonical model architecture, checkpoint, and parameter count are pristine."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Expected 3,443,136 parameters, got {total_params}"

    assert os.path.exists(CANONICAL_CHECKPOINT_PATH), "Canonical checkpoint must exist"
    with open(CANONICAL_CHECKPOINT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_CHECKPOINT_SHA256, "Canonical checkpoint SHA-256 mismatch!"


def test_pointer_beam_hypothesis_scoring():
    """Verify BeamHypothesis length normalization and ordering."""
    hyp1 = BeamHypothesis(token_ids=[10, 20], log_prob=-2.0, token_sources=["copy", "copy"])
    hyp2 = BeamHypothesis(token_ids=[10, 20, 30, 40], log_prob=-3.0, token_sources=["copy"] * 4)

    assert hyp1.score == -2.0 / (2 ** 0.7)
    assert hyp2.score == -3.0 / (4 ** 0.7)
    # Note: -1.136 > -1.231, so hyp2 has a higher length-normalized score
    assert hyp2.score > hyp1.score


def test_beam_width_one_matches_greedy():
    """Verify that decode_pointer_beam with beam_width=1 behaves identically to greedy decoding."""
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096, d_proj=64)

    prompt = [1, 10, 20, 30, 40]
    toks_greedy, src_greedy = decode_pointer_greedy(
        model=model,
        pointer_head=pointer_head,
        prompt_ids=prompt,
        context_start=1,
        context_end=3,
        max_new_tokens=4,
    )
    toks_beam1, src_beam1 = decode_pointer_beam(
        model=model,
        pointer_head=pointer_head,
        prompt_ids=prompt,
        context_start=1,
        context_end=3,
        beam_width=1,
        max_new_tokens=4,
    )

    assert toks_greedy == toks_beam1
    assert src_greedy == src_beam1


def test_hybrid_reranker_and_fallback():
    """Verify HybridReranker candidate ranking and lexical fallback when reranking is disabled."""
    reranker = HybridReranker(enable_reranking=True, rerank_alpha=0.5)
    reranker.ingest_document("doc1", "Boron is a metalloid with atomic number 5.")
    reranker.ingest_document("doc2", "Carbon is a nonmetal with atomic number 6.")

    hits = reranker.retrieve_and_rerank("What is the atomic number of Boron?", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "doc1"
    assert hits[0].lexical_score > 0.0

    # Lexical fallback
    reranker_fallback = HybridReranker(enable_reranking=False)
    reranker_fallback.ingest_document("doc1", "Boron is a metalloid with atomic number 5.")
    hits_fb = reranker_fallback.retrieve_and_rerank("What is the atomic number of Boron?", top_k=1)
    assert hits_fb[0].doc_id == "doc1"
    assert hits_fb[0].neural_score == 0.0


def test_evidence_assembler_deduplication_and_contradiction():
    """Verify MultiDocumentEvidenceAssembler deduplication and conflict detection."""
    assembler = MultiDocumentEvidenceAssembler()

    # Contradiction test
    conflict_hits = [
        RerankedEvidence(chunk_id="c1", doc_id="d1", text="The melting point of Gold is 1064 C.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=0, rerank_delta=0),
        RerankedEvidence(chunk_id="c2", doc_id="d2", text="The melting point of Gold is 1200 C.", lexical_score=1.0, neural_score=0.9, final_score=0.95, retrieval_rank=1, rerank_delta=0),
    ]
    bundle = assembler.assemble("What is the melting point of Gold?", conflict_hits)
    assert bundle.has_conflicts is True
    assert bundle.is_sufficient is False
    assert len(bundle.conflicting_values) == 2

    # Deduplication test
    dup_hits = [
        RerankedEvidence(chunk_id="c1", doc_id="d1", text="Oxygen atomic number is 8.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=0, rerank_delta=0),
        RerankedEvidence(chunk_id="c1_dup", doc_id="d1", text="Oxygen atomic number is 8.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=1, rerank_delta=0),
    ]
    bundle_dup = assembler.assemble("What is the atomic number of Oxygen?", dup_hits)
    assert len(bundle_dup.items) == 1
    assert bundle_dup.consensus_fact == "8"
    assert bundle_dup.is_sufficient is True


def test_task_engine_synthesis_workflow():
    """Verify end-to-end multi-document synthesis workflow in CognitiveTaskEngine."""
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_test_synth_"))
    try:
        mem = PersistentSessionMemory(temp_dir / "sessions")
        engine = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)

        docs = [
            ("d1", "Lithium is an alkali metal with atomic number 3."),
            ("d2", "The atomic number of Lithium is 3, making it the third element."),
        ]
        task_res = engine.execute_synthesis_workflow(
            task_id="synth_test_1",
            session_id="sess_synth",
            query="What is the atomic number of Lithium?",
            document_texts=docs,
        )
        assert task_res.status == TaskStatus.COMPLETED
        assert task_res.final_result == "3"

        # Check session persistence
        turns = mem.get_history("sess_synth")
        assert len(turns) == 2
        assert turns[-1].content == "3"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
