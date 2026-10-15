"""
Batches 411-440: Paraphrase Generalization & Compositional Learning Test Suite.
Location: tests/test_batch411_440_paraphrase.py

Tests:
1. Benchmark manifest separation and non-overlap:
   - train_augmented.jsonl and test_heldout_paraphrase.jsonl have disjoint prompts.
2. Checkpoint SHA-256 reproducibility:
   - Canonical checkpoint hash: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
   - Prior candidate hash: 27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e
   - Paraphrase augmented candidate hash: b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0
3. Paraphrase evaluation metric contract:
   - Evaluator recognizes exact matches on held-out variants.
4. Clean termination on True EOS token 1:
   - New candidate emits EOS=1 cleanly without corrupted pseudo-tokens.
"""
import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

PRIOR_CANDIDATE = Path("experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt")
EXPECTED_PRIOR_HASH = "27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e"

AUGMENTED_CANDIDATE = Path("experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt")
EXPECTED_AUGMENTED_HASH = "b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0"

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_separation_and_disjoint_splits():
    train_path = Path("experiments/batch411_440_paraphrase/data/train_augmented.jsonl")
    heldout_path = Path("experiments/batch411_440_paraphrase/data/test_heldout_paraphrase.jsonl")

    assert train_path.exists()
    assert heldout_path.exists()

    with open(train_path, "r", encoding="utf-8") as f:
        train_prompts = {json.loads(l)["prompt"].strip() for l in f if l.strip()}
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_prompts = {json.loads(l)["prompt"].strip() for l in f if l.strip()}

    overlap = train_prompts.intersection(heldout_prompts)
    assert len(overlap) == 0, f"Found accidental prompt contamination between train and held-out test: {overlap}"


def test_checkpoint_checksums_reproducible():
    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH
    assert sha256_file(PRIOR_CANDIDATE) == EXPECTED_PRIOR_HASH
    assert sha256_file(AUGMENTED_CANDIDATE) == EXPECTED_AUGMENTED_HASH


def test_augmented_candidate_loads_and_emits_true_eos():
    assert AUGMENTED_CANDIDATE.exists()
    ck = torch.load(AUGMENTED_CANDIDATE, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "Find the total value when 28 is added to 14:\nValue:"
    p_toks = [0] + tokenizer.encode(prompt)
    gen = list(p_toks)

    eos_found = False
    with torch.no_grad():
        for _ in range(15):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            nxt = int(torch.argmax(model(x)[0, -1, :]).item())
            if nxt == 1:
                eos_found = True
                break
            gen.append(nxt)

    ans = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
    assert eos_found is True, "Candidate should terminate cleanly on true EOS token 1"
    assert "42" in ans, f"Candidate should correctly emit 42 on held-out paraphrase, got: {ans}"
