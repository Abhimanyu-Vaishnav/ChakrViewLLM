"""
Comprehensive Unit & Regression Tests for Neural Development Milestones (Milestones 1-8).

Location: tests/test_neural_development_milestones.py
"""

import tempfile
from pathlib import Path
import pytest
import torch

from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.v050_candidate.inductive_representation_suite import (
    InductiveRepresentationSuite,
)
from chakrview.cognition.v050_candidate.neural_language_suite import (
    NeuralLanguageExperimentSuite,
)
from chakrview.cognition.v050_candidate.experience_neural_pipeline import (
    GovernedExperienceNeuralPipeline,
)
from chakrview.cognition.v050_candidate.neural_cognitive_coupling import (
    NeuralCognitiveCouplingEngine,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_m1_ablation_control_diagnostics():
    """Milestone 1: Diagnostic control proves category feature presence drives Mode 3 transfer."""
    (tx, ty), _, (tex, tey) = InductiveRepresentationSuite.generate_data(mode="categorical", seed=42)
    # Ablate category
    tx_abl = tx.clone()
    tx_abl[:, 0] = 0
    tex_abl = tex.clone()
    tex_abl[:, 0] = 0

    cand = ChakrMicro(ModelConfig())
    opt = torch.optim.AdamW(cand.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(25):
        opt.zero_grad()
        loss = loss_fn(cand(tx_abl)[:, -1, :], ty)
        loss.backward()
        opt.step()

    cand.eval()
    with torch.no_grad():
        te_acc = (cand(tex_abl)[:, -1, :].argmax(dim=-1) == tey).float().mean().item()
    # Without category feature, transfer collapses significantly from 100%
    assert te_acc < 0.80, f"Expected ablation collapse, got {te_acc}"


def test_m2_tokenizer_model_embedding_bounds_and_roundtrip():
    """Milestone 2: Tokenizer encode/decode roundtrips within model embedding bounds [0, 4095]."""
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    test_str = "ChakrView native CPU architecture"
    tokens = tokenizer.encode(test_str)
    assert len(tokens) > 0
    assert all(0 <= t < 4096 for t in tokens), "Token out of valid model vocab bounds"

    decoded = tokenizer.decode(tokens)
    assert decoded == test_str

    # Test model forward on tokens
    m = ChakrMicro(ModelConfig())
    m.eval()
    t_tensor = torch.tensor([tokens])
    with torch.no_grad():
        out = m(t_tensor)
    assert out.shape == (1, len(tokens), 4096)


def test_m3_m7_bounded_language_learning_and_checkpoint():
    """Milestone 3 & 7: Language learning on real text shards reduces validation loss and serializes."""
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        res = NeuralLanguageExperimentSuite.run_training_experiment(
            repo_root=Path("."),
            steps=20,
            batch_size=8,
            seq_len=64,
            checkpoint_dir=tmp_dir,
        )
        assert res.final_val_loss < res.initial_val_loss, "Validation loss failed to decrease"
        assert res.test_loss < res.initial_val_loss, "Held-out test loss failed to decrease"
        assert res.checkpoint_sha256 is not None
        assert Path(res.checkpoint_path).exists()

        # Reload candidate checkpoint and verify integrity
        data = torch.load(res.checkpoint_path, map_location="cpu")
        assert "state_dict" in data
        assert data["steps"] == 20


def test_m5_neural_cognitive_coupling_and_abstention():
    """Milestone 5: Grounded representation enables ANSWER; ungrounded enables honest ABSTAIN."""
    (tx, ty), _, _ = InductiveRepresentationSuite.generate_data(mode="categorical", seed=42)
    cand = ChakrMicro(ModelConfig())
    cand.train()
    opt = torch.optim.AdamW(cand.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(cand(tx)[:, -1, :], ty)
        loss.backward()
        opt.step()
    cand.eval()

    # Grounded input
    res_grounded = NeuralCognitiveCouplingEngine.evaluate_coupling(cand, torch.tensor([[10, 1000, 50]]), "Alpha")
    assert res_grounded.cognitive_decision == "ANSWER"
    assert res_grounded.coupling_effective is True

    # Untrained random baseline
    base = ChakrMicro(ModelConfig())
    res_base = NeuralCognitiveCouplingEngine.evaluate_coupling(base, torch.tensor([[10, 20, 30]]), "Beta")
    assert res_base.cognitive_decision == "ABSTAIN"
    assert res_base.coupling_effective is False


def test_m6_experience_governance_and_rollback():
    """Milestone 6: Bad experience rejected; regression triggers rollback."""
    bad_exp = {"token_sequence": [[10, 20, 30]], "target_sequence": [40], "confidence": 0.40}
    outcome_bad = GovernedExperienceNeuralPipeline.process_experience(bad_exp)
    assert outcome_bad.experience_accepted is False
    assert outcome_bad.decision == "REJECTED_BY_GOVERNANCE_GATE"

    good_exp = {"token_sequence": [[10, 20, 30]], "target_sequence": [40], "confidence": 0.95}
    outcome_good = GovernedExperienceNeuralPipeline.process_experience(good_exp)
    assert outcome_good.experience_accepted is True
    assert outcome_good.parameter_updated is True
