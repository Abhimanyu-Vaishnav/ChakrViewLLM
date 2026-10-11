"""
Unit and regression tests for Batch 41: Evaluation Reliability, Runtime/Training Alignment, and Baseline Invariance.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136


def test_canonical_baseline_and_checkpoint_hash():
    """Verifies that fresh canonical ChakrMicro initialization matches expected hash and params."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    digest = hasher.hexdigest()
    assert digest == EXPECTED_CANONICAL_HASH


def test_chatml_subword_tokenization_properties():
    """Verifies that ChatML tags tokenize to deterministic subwords within frozen 4096 vocabulary."""
    tok, meta = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    start_ids = tok.encode(ConversationTemplate.START_TAG)
    end_ids = tok.encode(ConversationTemplate.END_TAG)

    assert len(start_ids) == 8
    assert len(end_ids) == 8
    assert all(0 <= t < 4096 for t in start_ids)
    assert all(0 <= t < 4096 for t in end_ids)
    assert tok.decode(start_ids) == ConversationTemplate.START_TAG
    assert tok.decode(end_ids) == ConversationTemplate.END_TAG


def test_conversation_template_alignment_and_masking(tmp_path):
    """Verifies that ChatML formatted prompts mask all user/system tokens and only supervise assistant tokens."""
    tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")

    user_text = "What is Pre-RMSNorm?"
    assistant_text = "Pre-RMSNorm normalizes inputs before self-attention."

    messages = [ChatMessage(role=Role.USER, content=user_text)]
    prompt_str = ConversationTemplate.format_conversation(messages, add_generation_prompt=True)
    response_str = f"{assistant_text}{ConversationTemplate.END_TAG}\n"

    dummy_file = tmp_path / "dummy.jsonl"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt_str, "response": response_str}) + "\n")

    ds = SFTDataset(dummy_file, tok, max_seq_len=512)
    inp, tgt = ds.encode_example(prompt_str, response_str)

    p_toks = tok.encode(prompt_str.strip())
    r_toks = tok.encode(response_str.strip())

    # Prompt portion must be completely masked (-100)
    assert tgt[:len(p_toks)] == [-100] * len(p_toks)
    # Response portion must match r_toks
    assert tgt[len(p_toks):-1] == r_toks
    # Final token must be EOS
    assert tgt[-1] == 1


def test_batch41_checkpoint_reload_and_probe_determinism():
    """Verifies that the aligned Batch 41 checkpoint reloads and reproduces evaluation scores."""
    ckpt_path = Path("experiments/batch41_evaluation_git/reproducibility/sft_aligned_checkpoint_step_20.pt")
    if not ckpt_path.exists():
        pytest.skip("Aligned checkpoint not yet created")

    summary_path = Path("experiments/batch41_evaluation_git/reproducibility/eval_summary.json")
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(ckpt_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == summary["checkpoint"]["sha256"]

    # Reload checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    prompt = "<|im_start|>user\nWhat is Pre-RMSNorm?<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens = [0] + tok.encode(prompt)
    generated = list(prompt_tokens)
    for _ in range(40):
        with torch.no_grad():
            logits = model(torch.tensor([generated], dtype=torch.long))
            next_tok = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_tok)
            if next_tok == 1:
                break
    text = tok.decode(generated[len(prompt_tokens):])
    saved_text = summary["evaluation"]["after"]["probes"][2]["generated_text"]
    assert text == saved_text
