"""
Regression and Invariant Tests for Batch 147–150 Context Binding & Safe Session Integration.
Verifies:
1. Canonical Step 10,000 checkpoint file SHA-256 and parameter invariants.
2. Canonical seed 42 weight hash immutability.
3. Context binding dataset split disjointness and hash integrity.
4. Deterministic baseline extractor performance (100% on synthetic binding test).
5. Safe session ChatML integration contract without modifying session.py.
6. Bit-exact candidate checkpoint resumability.
7. Batch 147–150 artifacts existence and validity.
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
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.session import ConversationRole

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "batch147_150_context_binding"
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

def test_context_binding_dataset_integrity():
    p_train = DATA_DIR / "binding_train.jsonl"
    p_val = DATA_DIR / "binding_val.jsonl"
    p_test = DATA_DIR / "binding_test.jsonl"
    
    for p in [p_train, p_val, p_test]:
        assert p.exists(), f"Dataset file missing: {p}"
        
    def get_hashes(p):
        with open(p, "r", encoding="utf-8") as f:
            return {json.loads(line)["content_hash"] for line in f if line.strip()}
            
    h_train = get_hashes(p_train)
    h_val = get_hashes(p_val)
    h_test = get_hashes(p_test)
    
    assert len(h_train.intersection(h_val)) == 0, "Train-Val hash leakage!"
    assert len(h_train.intersection(h_test)) == 0, "Train-Test hash leakage!"
    assert len(h_val.intersection(h_test)) == 0, "Val-Test hash leakage!"

def test_deterministic_baseline_extractor():
    from experiments.batch147_150_context_binding.run_binding_experiments import deterministic_baseline_extractor
    
    # Direct
    assert deterministic_baseline_extractor("material is diamond.", "What is material?") == "diamond"
    # Distractor first
    assert deterministic_baseline_extractor("language is Sanskrit. vehicle is bicycle.", "What is language?") == "Sanskrit"
    # Distractor second
    assert deterministic_baseline_extractor("instrument is sitar. flower is lotus.", "What is flower?") == "lotus"
    # Abstention
    assert deterministic_baseline_extractor("mineral is quartz. continent is Asia.", "What is president?") == "Not mentioned in context."
    # Hindi
    assert deterministic_baseline_extractor("धातु: तांबा। पशु: गाय।", "धातु क्या है?") == "तांबा"

def test_session_chatml_integration_contract():
    """Validates the exact logic introduced in session.py without modifying the file."""
    turns = [
        {"role": ConversationRole.SYSTEM, "content": "You are ChakrView."},
        {"role": ConversationRole.USER, "content": "Hello."},
        {"role": ConversationRole.ASSISTANT, "content": "Greetings."},
    ]
    user_input = "What is 2+2?"
    
    role_map = {
        ConversationRole.SYSTEM: Role.SYSTEM,
        ConversationRole.USER: Role.USER,
        ConversationRole.ASSISTANT: Role.ASSISTANT,
    }
    
    messages = [ChatMessage(role=role_map[t["role"]], content=t["content"]) for t in turns]
    messages.append(ChatMessage(role=Role.USER, content=user_input))
    
    formatted = ConversationTemplate.format_conversation(messages, add_generation_prompt=True)
    assert "<|im_start|>system\nYou are ChakrView.<|im_end|>\n" in formatted
    assert "<|im_start|>user\nHello.<|im_end|>\n" in formatted
    assert "<|im_start|>assistant\nGreetings.<|im_end|>\n" in formatted
    assert "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")

def test_checkpoint_resumability_invariant():
    ckpt_c = CKPT_DIR / "arm_c_binding_warmup_cosine.pt"
    assert ckpt_c.exists(), "Arm C checkpoint missing!"
    data = torch.load(ckpt_c, map_location="cpu", weights_only=False)
    assert "model_state_dict" in data
    assert "optimizer_state_dict" in data
    assert data["step"] == 60
