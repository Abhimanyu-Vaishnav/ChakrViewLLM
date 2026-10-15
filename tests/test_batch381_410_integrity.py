"""
Batches 381-410: Architecture, Checkpoint & Evaluation Integrity Test Suite.
Location: tests/test_batch381_410_integrity.py

Tests:
1. Exact parameter count verification:
   - Unique model parameters = 3,443,136
   - Serialized state_dict entries = 4,229,568 (reconciling tied LMHead weight)
2. Checkpoint SHA-256 reproducibility:
   - Canonical step 10,000 hash: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
   - Candidate step 150 (batch311-350) hash: b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759
3. Corrected Candidate step 150 checkpoint validity and true EOS emission.
4. Loss masking shift alignment invariant: shift_x[len(prompt)-1] predicts target[0].
"""
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

STEP150_CHECKPOINT = Path("experiments/batch311_350_neural_generalization/checkpoints/candidate_step150.pt")
EXPECTED_STEP150_HASH = "b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759"

CORRECTED_CHECKPOINT = Path("experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_parameter_count_discrepancy_resolved():
    ck = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]

    # In serialized checkpoint: both embedding.weight and lm_head.weight are saved
    serialized_elements = sum(t.numel() for t in sd.values())
    assert serialized_elements == 4_229_568, f"Expected 4229568 serialized elements, got {serialized_elements}"

    # In instantiated model: weight tying is enforced
    model = ChakrMicro(ck["config"])
    model.load_state_dict(sd)

    unique_params = sum(p.numel() for p in set(model.parameters()))
    assert unique_params == 3_443_136, f"Expected 3443136 unique parameters, got {unique_params}"

    counts = model.count_parameters()
    assert counts["total_parameters"] == 3_443_136
    assert counts["output_head_unique"] == 0


def test_checkpoint_hashes_reproducible():
    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH
    assert sha256_file(STEP150_CHECKPOINT) == EXPECTED_STEP150_HASH


def test_corrected_candidate_checkpoint_and_true_eos():
    assert CORRECTED_CHECKPOINT.exists()
    ck = torch.load(CORRECTED_CHECKPOINT, map_location="cpu", weights_only=False)
    assert ck["name"] == "candidate_step150_corrected_eos"

    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "Question: What is 14 plus 28?\nAnswer:"
    p_toks = [0] + tokenizer.encode(prompt)
    gen = list(p_toks)

    eos_found = False
    with torch.no_grad():
        for _ in range(15):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            logits = model(x)
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            if nxt == 1:
                eos_found = True
                break
            gen.append(nxt)

    ans = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
    assert eos_found is True, "Candidate should terminate cleanly on true EOS token 1"
    assert "42" in ans, f"Candidate should correctly emit 42, got: {ans}"


def test_loss_masking_mathematical_shift_alignment():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    p_text = "Prompt test query"
    r_text = " Target response"

    p_toks = [0] + tokenizer.encode(p_text)
    r_toks = tokenizer.encode(r_text) + [1]

    input_ids = p_toks + r_toks
    labels = [-100] * len(p_toks) + r_toks

    x = torch.tensor([input_ids], dtype=torch.long)
    y = torch.tensor([labels], dtype=torch.long)

    shift_x = x[:, :-1]
    shift_y = y[:, 1:]

    # Prompt boundary prediction:
    # shift_x at index (len(p_toks) - 1) is the last token of prompt
    # shift_y at index (len(p_toks) - 1) is labels[len(p_toks)], which is r_toks[0]
    idx_first_target = len(p_toks) - 1
    assert shift_y[0, idx_first_target].item() == r_toks[0]
    assert shift_y[0, idx_first_target - 1].item() == -100
