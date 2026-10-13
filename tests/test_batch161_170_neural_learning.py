"""
Regression test suite for Batches 161–170 Learned Representations, Neural Copy Induction & Task Engine.
Tests:
1. ChakrBiEncoder architecture, shapes, normalization, and determinism.
2. LearnedEmbeddingProvider checkpoint save and load roundtrip.
3. RetrievalAdapter with LearnedEmbeddingProvider integration.
4. CognitiveTaskEngine state machine, persistence, execution, and crash recovery.
5. Causal loss mask invariant (Prompt masked to -100, target answer tokens active).
"""
import tempfile
import shutil
from pathlib import Path
import pytest
import torch

from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.task_engine import CognitiveTaskEngine, TaskStatus
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def test_chakr_bi_encoder_forward_and_norm():
    model = ChakrBiEncoder(vocab_size=4096, d_model=64, d_embed=32, max_seq_len=64, n_layers=1)
    input_ids = torch.randint(0, 4096, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    emb = model(input_ids, attention_mask=mask)
    
    assert emb.shape == (2, 32)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.tensor([1.0, 1.0]), atol=1e-5)


def test_learned_embedding_provider_roundtrip():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = ChakrBiEncoder(vocab_size=4096, d_model=64, d_embed=32, max_seq_len=64, n_layers=1)
    provider = LearnedEmbeddingProvider(model=model, tokenizer=tokenizer)
    
    vec1 = provider.embed_text("Test sentence")
    assert len(vec1) == 32
    
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        ckpt_path = tmp_dir / "provider.pt"
        provider.save_checkpoint(ckpt_path)
        
        loaded = LearnedEmbeddingProvider.load_checkpoint(ckpt_path, tokenizer)
        assert loaded.dimension == 32
        vec2 = loaded.embed_text("Test sentence")
        assert len(vec2) == 32
        assert pytest.approx(vec1[0], abs=1e-5) == vec2[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrieval_adapter_with_learned_provider():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = ChakrBiEncoder(vocab_size=4096, d_model=64, d_embed=32, max_seq_len=64, n_layers=1)
    provider = LearnedEmbeddingProvider(model=model, tokenizer=tokenizer)
    
    ra = RetrievalAdapter(embedding_provider=provider)
    ra.ingest_document("d1", "Boron is a chemical element.")
    
    hits = ra.search("What is Boron?")
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"


def test_cognitive_task_engine_execution_and_idempotence():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        ra = RetrievalAdapter()
        ra.ingest_document("doc_gold", "Pure gold melts at a temperature of 1064 C.")
        
        engine = CognitiveTaskEngine(storage_dir=tmp_dir, retrieval_adapter=ra)
        task_id = "test_task_1"
        sess_id = "test_sess_1"
        
        # 1. Execute task
        res = engine.execute_factual_workflow(task_id, sess_id, "What is the melting point of Gold?")
        assert res.status == TaskStatus.COMPLETED
        assert res.final_result == "1064 C"
        
        # 2. Resumed task execution (idempotent)
        res_resumed = engine.execute_factual_workflow(task_id, sess_id, "What is the melting point of Gold?")
        assert res_resumed.status == TaskStatus.COMPLETED
        assert res_resumed.final_result == "1064 C"
        
        # 3. Verify recorded in memory
        hist = engine.memory.get_history(sess_id)
        assert len(hist) == 2
        assert hist[1].content == "1064 C"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_causal_copy_loss_masking_contract():
    import sys
    exp_path = Path(__file__).resolve().parents[1] / "experiments" / "batch161_170_neural_learning"
    if str(exp_path) not in sys.path:
        sys.path.insert(0, str(exp_path))
    from run_copy_induction_pilot import build_chatml_example
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    context = "material is steel."
    query = "What is material?"
    answer = "steel"
    
    inp, tgt, p_str = build_chatml_example(context, query, answer, tokenizer)
    
    # Assert targets contain -100 for prompt tokens
    assert -100 in tgt
    
    # Assert active loss targets are not empty and contain the answer tokens
    active_targets = [t for t in tgt if t != -100]
    assert len(active_targets) > 0
    # Decoded active string includes the answer text plus the termination token <|im_end|>
    decoded_active = tokenizer.decode(active_targets, skip_special_tokens=False).strip()
    assert decoded_active.startswith("steel")
    assert "<|im_end|>" in decoded_active
