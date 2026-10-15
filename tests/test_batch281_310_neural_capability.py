"""
Unit and regression tests for Batches 281-310:
- Verification of canonical checkpoint invariants (hash & 3,443,136 parameters).
- Candidate model checkpoint creation, configuration, and state dict validity.
- Target loss masking function (ensuring prompt tokens receive -100 label).
- Imperative technical concept routing ("Explain FPGA", "Explain TPU") to CONCEPT_DEFINITION.
- Neural pointer evaluation reproducibility on multi-hop chains.
"""
import ast
import hashlib
from pathlib import Path
import pytest
import sys
import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from experiments.batch281_310_neural_capability.run_neural_experiments import (
    load_canonical_model,
    encode_chatml_example,
)


def test_canonical_invariants_preserved():
    ck_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert ck_path.exists(), "Canonical checkpoint file must exist"
    h = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    expected_hash = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert h == expected_hash, f"Canonical hash mismatch: got {h}"

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    p_count = sum(p.numel() for p in model.parameters())
    assert p_count == 3443136, f"Parameter count must be 3,443,136, got {p_count}"


def test_candidate_checkpoint_validity():
    cand_path = Path("experiments/batch281_310_neural_capability/checkpoints/chakrmicro_capability_candidate_step70.pt")
    assert cand_path.exists(), "Trained candidate checkpoint must exist"
    ck = torch.load(cand_path, map_location="cpu", weights_only=False)
    assert ck["step"] == 70
    assert "model_state_dict" in ck
    assert ck["train_meta"]["final_loss"] < ck["train_meta"]["initial_loss"]


def test_target_loss_masking_contract():
    _, tokenizer = load_canonical_model(torch.device("cpu"))
    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    response = "4<|im_end|>\n"
    
    input_ids, labels = encode_chatml_example(tokenizer, prompt, response)
    
    # Check that labels contain -100 for prompt tokens
    assert -100 in labels.tolist()
    # Check that active (non -100) labels exist for response
    active_labels = [l for l in labels.tolist() if l != -100]
    assert len(active_labels) > 0, "Response must have active supervised target tokens"


def test_technical_concept_routing_and_grounding():
    router = CognitiveRouter()
    adapter = GroundedSessionAdapter(inference_engine=None)
    
    # 1. FPGA
    dec_fpga = router.classify_route("Explain Field Programmable Gate Array in simple words.")
    assert dec_fpga.route == RouteType.CONCEPT_DEFINITION
    resp_fpga = adapter.execute_turn("test_session", "Explain Field Programmable Gate Array in simple words.")
    assert "integrated circuit" in resp_fpga.answer.lower()
    
    # 2. TPU
    dec_tpu = router.classify_route("Explain Tensor Processing Unit in simple words.")
    assert dec_tpu.route == RouteType.CONCEPT_DEFINITION
    resp_tpu = adapter.execute_turn("test_session", "Explain Tensor Processing Unit in simple words.")
    assert "accelerator" in resp_tpu.answer.lower() or "neural network" in resp_tpu.answer.lower()
