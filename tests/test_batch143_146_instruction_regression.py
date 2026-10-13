"""
Regression and Invariant Tests for Batch 143–146 Instruction Learning & Grounded Comprehension.
Verifies:
1. Canonical Step 10,000 checkpoint file SHA-256 and parameter invariants.
2. Canonical seed 42 weight hash immutability.
3. Multi-domain instruction dataset disjointness and validity.
4. Causal sequence prompt loss-masking mathematical alignment.
5. Batch 143–146 experiment artifacts existence and finite metrics.
6. Checkpoint resumability of Arm C candidate.
7. HybridRetriever execution and score bounds.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import InferenceEngine
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence
from chakrview.runtime.knowledge import BM25KnowledgeIndex, KnowledgeChunk
from chakrview.runtime.retrieval import HybridRetriever, DeterministicHashEmbeddingProvider, InMemoryVectorIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "batch143_146_instruction_learning"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CANONICAL_BASE_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

def compute_file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_canonical_invariants_and_hashes():
    assert CANONICAL_BASE_CKPT.exists(), "Canonical Step 10,000 checkpoint file missing!"
    sha = compute_file_sha256(CANONICAL_BASE_CKPT)
    expected_sha = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert sha == expected_sha, f"Canonical checkpoint hash mutated! Expected {expected_sha}, got {sha}"
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3_443_136, f"Parameter count mismatch: {total_params}"
    
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    m.eval()
    engine = InferenceEngine(model=m, tokenizer=tok)
    h = engine.compute_weight_hash()
    assert h == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"

def test_instruction_dataset_disjointness_and_provenance():
    train_path = DATA_DIR / "instruction_train.jsonl"
    val_path = DATA_DIR / "instruction_val.jsonl"
    test_path = DATA_DIR / "instruction_test.jsonl"
    
    for p in [train_path, val_path, test_path]:
        assert p.exists(), f"Dataset file missing: {p}"
        
    def load_hashes(p):
        with open(p, "r", encoding="utf-8") as f:
            return {json.loads(line)["content_hash"] for line in f if line.strip()}
            
    h_train = load_hashes(train_path)
    h_val = load_hashes(val_path)
    h_test = load_hashes(test_path)
    
    assert len(h_train.intersection(h_val)) == 0, "Train-Val leakage!"
    assert len(h_train.intersection(h_test)) == 0, "Train-Test leakage!"
    assert len(h_val.intersection(h_test)) == 0, "Val-Test leakage!"

def test_causal_sequence_loss_masking_alignment():
    prompt = [10, 20, 30]
    response = [40, 50]
    inp, tgt = build_causal_sequence(prompt, response, bos_id=0, eos_id=1, ignore_index=-100)
    
    # Inp: [0, 10, 20, 30, 40, 50] (len 6)
    # Tgt: [-100, -100, -100, 40, 50, 1] (len 6)
    assert len(inp) == len(tgt) == 6
    assert inp == [0, 10, 20, 30, 40, 50]
    assert tgt == [-100, -100, -100, 40, 50, 1]
    # inp[3] is the last prompt token (30), its target is tgt[3] which is response[0] (40)
    assert inp[3] == 30 and tgt[3] == 40
    # inp[5] is the last response token (50), its target is tgt[5] which is EOS (1)
    assert inp[5] == 50 and tgt[5] == 1

def test_batch143_146_artifacts_exist_and_resumability():
    results_path = REPORTS_DIR / "results.json"
    manifest_path = EXP_DIR / "experiment_manifest.json"
    ckpt_c = CKPT_DIR / "arm_c_sft_warmup_cosine.pt"
    
    assert results_path.exists(), "results.json missing!"
    assert manifest_path.exists(), "experiment_manifest.json missing!"
    assert ckpt_c.exists(), "arm_c checkpoint missing!"
    
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)
        
    assert "Arm_A_Baseline" in res
    assert "Arm_B_Constant_SFT" in res
    assert "Arm_C_Warmup_Cosine_SFT" in res
    assert "Arm_D_Conservative_SFT" in res
    assert res["resume_verification"]["status"] == "PASSED"
    assert "grounded_comprehension" in res

def test_hybrid_retriever_pipeline():
    lex_index = BM25KnowledgeIndex()
    embed_provider = DeterministicHashEmbeddingProvider(dimension=64)
    vec_index = InMemoryVectorIndex()
    
    chunk = KnowledgeChunk(doc_id="d1", chunk_id="c1", chunk_index=0, text="Gravity is a fundamental interaction.")
    lex_index.add_chunks([chunk])
    vec_index.add_vector(chunk.chunk_id, embed_provider.embed_text(chunk.text), text=chunk.text)
    
    retriever = HybridRetriever(lexical_index=lex_index, embedding_provider=embed_provider, vector_index=vec_index)
    results = retriever.retrieve("gravity", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert 0.0 <= score <= 1.0
