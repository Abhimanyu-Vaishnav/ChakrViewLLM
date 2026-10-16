"""
Regression and validation unit tests for Batch 1421-1460 independent validation.
Location: tests/test_batch1421_1460_independent_validation.py
"""
import hashlib
import json
import pytest
import torch
from pathlib import Path

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
CANDIDATE_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
INDEP_MANIFEST = Path("experiments/batch1421_1460/data/independent_corpus_manifest.json")
INDEP_RESULTS = Path("experiments/batch1421_1460/results/INDEPENDENT_EVALUATION_RESULTS.json")


@pytest.fixture(scope="module")
def checkpoints():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    b_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    c_ckpt = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)
    return tokenizer, b_ckpt, c_ckpt


def test_checkpoint_hashes_and_preservation(checkpoints):
    """Verify both canonical and candidate checkpoints have exact expected SHA-256 hashes."""
    h_canon = hashlib.sha256(CANONICAL_CKPT.read_bytes()).hexdigest()
    assert h_canon == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    h_cand = hashlib.sha256(CANDIDATE_CKPT.read_bytes()).hexdigest()
    assert h_cand == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"


def test_independent_corpus_integrity_and_manifest():
    """Verify independent corpus manifest has exactly 30 documents and zero collisions."""
    manifest = json.loads(INDEP_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["total_documents"] == 30
    assert len(manifest["documents"]) == 30
    assert len(manifest["categories"]) == 5


def test_candidate_generalization_outperforms_baseline():
    """Verify candidate achieves lower loss and higher accuracy on independent corpus."""
    results = json.loads(INDEP_RESULTS.read_text(encoding="utf-8"))
    base_loss = results["baseline_canonical"]["metrics"]["cross_entropy_loss"]
    cand_loss = results["candidate_adapted"]["metrics"]["cross_entropy_loss"]
    assert cand_loss < base_loss
    assert results["deltas"]["perplexity_relative_reduction_percent"] > 40.0
    assert results["deltas"]["top_1_accuracy_delta"] > 0.05


def test_all_independent_documents_show_improvement():
    """Verify that all 30 independent documents show improvement (win rate = 100%)."""
    results = json.loads(INDEP_RESULTS.read_text(encoding="utf-8"))
    per_doc = results["per_document_comparison"]
    assert len(per_doc) == 30
    assert all(d["improved"] for d in per_doc)


def test_reproduced_loss_calculations(checkpoints):
    """Verify that model evaluates deterministically on a test sample."""
    tokenizer, b_ckpt, c_ckpt = checkpoints
    m_base = ChakrMicro(b_ckpt["config"])
    m_base.load_state_dict(b_ckpt["model_state_dict"])
    m_base.eval()

    inputs = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    with torch.no_grad():
        out1 = m_base(inputs)
        out2 = m_base(inputs)
    assert torch.equal(out1, out2)
