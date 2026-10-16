"""
Regression test suite for Batches 1181–1220:
Domain-Adaptive Retrieval Learning, Safe Checkpoint Loading, and Invariant Preservation.
"""
from pathlib import Path
import pytest
import torch

from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant
from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import (
    CachedEmbeddingProvider,
    DeterministicHashEmbeddingProvider,
    InMemoryVectorIndex
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DOMAIN_ADAPTIVE_CKPT = Path("experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

def test_canonical_checkpoint_sha256_unmodified():
    import hashlib
    h = hashlib.sha256(CANONICAL_CKPT.read_bytes()).hexdigest()
    assert h == EXPECTED_CANONICAL_SHA, f"Canonical checkpoint hash altered! Got {h}"

def test_safe_checkpoint_loading_and_metadata():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    provider = LearnedEmbeddingProvider.load_checkpoint(DOMAIN_ADAPTIVE_CKPT, tokenizer=tokenizer)
    assert provider.dimension == 64
    assert provider.provider_id == "domain_adaptive_bi_encoder_v1"
    
    # Test safe handling on corrupt/missing checkpoint
    with pytest.raises(FileNotFoundError):
        LearnedEmbeddingProvider.load_checkpoint(Path("non_existent_ckpt.pt"), tokenizer=tokenizer)

def test_bm25_default_retriever_preservation():
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    # Ensure BM25 remains default when no neural provider is specified
    assert asst.embedding_provider is not None
    assert isinstance(asst.embedding_provider, (DeterministicHashEmbeddingProvider, CachedEmbeddingProvider))

def test_opt_in_domain_adaptive_provider():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    provider = LearnedEmbeddingProvider.load_checkpoint(DOMAIN_ADAPTIVE_CKPT, tokenizer=tokenizer)
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=provider,
        extract_concise_span=True,
        use_embedding_cache=True
    )
    asst.ingest_document(CORPUS_DIR / "network_encryption_transit.md")
    
    resp = asst.ask("What symmetric cipher algorithm is used with TLS 1.3?")
    assert resp.decision == "ANSWER"
    assert "AES-256-GCM" in resp.answer

def test_exact_numeric_and_distractor_retrieval():
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    asst.ingest_document(CORPUS_DIR / "compute_quantization.md")
    asst.ingest_document(CORPUS_DIR / "compute_accelerator_cuda.md")
    
    resp_cuda = asst.ask("What minimum compute capability is required for CUDA acceleration?")
    assert "7.5" in resp_cuda.answer
    
    resp_quant = asst.ask("What integer precision is used for weight quantization?")
    assert "INT8" in resp_quant.answer

def test_honest_abstention_on_unindexed_domain():
    asst = LocalKnowledgeAssistant(
        storage_dir=None,
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider=None,
        extract_concise_span=True
    )
    asst.ingest_document(CORPUS_DIR / "compute_arch_core.md")
    resp = asst.ask("What is the mass of the exoplanet Proxima Centauri b?")
    assert resp.decision == "ABSTAIN"
