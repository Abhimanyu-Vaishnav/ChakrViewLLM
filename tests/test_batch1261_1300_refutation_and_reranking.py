"""
Unit and Regression Test Suite for Batches 1261–1300:
Evidence-Governed Refutation and Hybrid Neural Reranking Invariants.
"""
from pathlib import Path
import pytest
import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant
from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DOMAIN_ADAPTIVE_CKPT = Path("experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt")
EXPECTED_DOMAIN_ADAPTIVE_SHA = "46543f36e6b172b3584654728c5166365d9761bbe587b5cfd10338a01218e41a"
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

def test_checkpoint_hashes_unmodified():
    import hashlib
    h_canon = hashlib.sha256(CANONICAL_CKPT.read_bytes()).hexdigest()
    assert h_canon == EXPECTED_CANONICAL_SHA, f"Canonical checkpoint altered! Got {h_canon}"
    
    h_dense = hashlib.sha256(DOMAIN_ADAPTIVE_CKPT.read_bytes()).hexdigest()
    assert h_dense == EXPECTED_DOMAIN_ADAPTIVE_SHA, f"Domain adaptive checkpoint altered! Got {h_dense}"

def test_evidence_grounded_refutation_on_numerical_premise():
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_document(CORPUS_DIR / "compute_arch_core.md")
    
    # Query asserts 128 attention heads, doc establishes 6 query heads
    resp = asst.ask("Why does ChakrMicro V0.6 require 128 attention heads on CPU?")
    assert resp.decision == "REFUTE"
    assert "Contradiction detected" in resp.answer
    assert "compute_arch_core.md" in resp.answer
    assert len(resp.citations) == 1

def test_evidence_grounded_refutation_on_prohibition_premise():
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_document(CORPUS_DIR / "security_sandbox_isolation.md")
    
    # Query asserts sudo allowed, doc establishes forbidden
    resp = asst.ask("Why does ChakrView allow arbitrary sudo execution in production?")
    assert resp.decision == "REFUTE"
    assert "Contradiction detected" in resp.answer
    assert "security_sandbox_isolation.md" in resp.answer

def test_unsupported_claim_triggers_abstention_not_refutation():
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_document(CORPUS_DIR / "security_auth_tokens.md")
    
    # Missing evidence must trigger ABSTAIN, not REFUTE
    resp = asst.ask("Why are database tokens configured to expire after 100 years?")
    assert resp.decision == "ABSTAIN"
    assert "could not find sufficient verified evidence" in resp.answer.lower()

def test_factual_query_retains_answer_decision():
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_document(CORPUS_DIR / "compute_accelerator_cuda.md")
    
    resp = asst.ask("What minimum compute capability is required for CUDA acceleration?")
    assert resp.decision == "ANSWER"
    assert "7.5" in resp.answer
    assert len(resp.citations) == 1

def test_bm25_candidate_generation_recall_at_10():
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    for f in sorted(CORPUS_DIR.glob("*.md")):
        asst.ingest_document(f)
        
    res = asst.retriever.lexical_index.search("What is the parameter count of ChakrMicro V0.6?", top_k=10)
    sources = [Path(chunk.doc_id).name for chunk, score in res]
    assert "compute_arch_core.md" in sources
