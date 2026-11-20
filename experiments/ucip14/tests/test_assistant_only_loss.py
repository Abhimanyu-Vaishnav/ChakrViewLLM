"""
Deterministic Token-Level and Gradient-Level Verification of Assistant-Only Masked Loss.
Verifies Phase 1 requirements:
1. BOS is correctly positioned at index 0.
2. Prompt tokens receive target label -100 and zero loss / zero logit gradients.
3. First assistant response token receives active loss and gradients.
4. Subsequent assistant response tokens receive active loss and gradients.
5. EOS receives active loss and gradients.
6. Padding tokens receive target label -100 and zero loss / zero logit gradients.
7. Strict causal alignment: input_seq[i] predicts target_seq[i], with zero off-by-one error.
8. Verified for both single-turn and multi-turn sequences.
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import build_causal_sequence


@pytest.fixture(scope="module")
def tokenizer_artifacts():
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    return tok


def test_01_single_turn_token_alignment(tokenizer_artifacts):
    tok = tokenizer_artifacts
    p_str = "User: Hi"
    r_str = "Hello"
    p_toks = tok.encode(p_str)
    r_toks = tok.encode(r_str)
    
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    assert inp[0] == 0, "BOS must be at index 0"
    assert inp[1:1+len(p_toks)] == p_toks, "Prompt tokens must immediately follow BOS"
    assert inp[1+len(p_toks):] == r_toks, "Response tokens must follow prompt tokens"
    
    # Prompt masking check
    for idx in range(len(p_toks)):
        assert tgt[idx] == -100, f"Prompt target at index {idx} must be masked (-100)"
        
    # Final prompt token index predicts first response token
    last_prompt_idx = len(p_toks)
    assert inp[last_prompt_idx] == p_toks[-1]
    assert tgt[last_prompt_idx] == r_toks[0]
    
    # Subsequent response tokens
    for i in range(1, len(r_toks)):
        curr_idx = last_prompt_idx + i
        assert inp[curr_idx] == r_toks[i-1]
        assert tgt[curr_idx] == r_toks[i]
        
    # Last token in inp predicts EOS
    last_idx = len(inp) - 1
    assert inp[last_idx] == r_toks[-1]
    assert tgt[last_idx] == 1, "Final target must be EOS (id=1)"


def test_02_gradient_flow_and_loss_isolation(tokenizer_artifacts):
    tok = tokenizer_artifacts
    p_str = "User: Hi"
    r_str = "Hello"
    p_toks = tok.encode(p_str)
    r_toks = tok.encode(r_str)
    
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    # Add 2 padding tokens
    pad_id = 2
    inp_padded = inp + [pad_id, pad_id]
    tgt_padded = tgt + [-100, -100]
    
    x = torch.tensor([inp_padded], dtype=torch.long)
    y = torch.tensor([tgt_padded], dtype=torch.long)
    
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.train()
    model.zero_grad()
    
    logits = model(x)
    logits.retain_grad()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, 4096), y.view(-1))
    loss.backward()
    
    grad_norms = logits.grad.abs().sum(dim=-1)[0]
    
    # Positions 0 to len(p_toks)-1 are prompt positions predicting -100
    for idx in range(len(p_toks)):
        assert grad_norms[idx].item() == 0.0, f"Prompt position {idx} must receive 0 gradient"
        
    # Active response predicting positions:
    # From last prompt token (predicting r_toks[0]) through last response token (predicting EOS).
    # Exactly len(r_toks) tokens + 1 EOS token = len(r_toks) + 1 active loss positions!
    active_start = len(p_toks)
    active_end = active_start + len(r_toks) + 1
    for idx in range(active_start, active_end):
        assert grad_norms[idx].item() > 0.0, f"Response position {idx} must receive positive gradient"
        
    # Padding positions must receive 0 gradient
    for idx in range(active_end, len(inp_padded)):
        assert grad_norms[idx].item() == 0.0, f"Padding position {idx} must receive 0 gradient"


def test_03_multi_turn_causal_alignment(tokenizer_artifacts):
    tok = tokenizer_artifacts
    prompt = "User: What is my name?\nAssistant: Alice\nUser: Repeat it.\nAssistant: "
    response = "Alice"
    p_toks = tok.encode(prompt)
    r_toks = tok.encode(response)
    
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    assert len(inp) == len(tgt)
    assert inp[0] == 0
    # First len(p_toks) targets masked
    assert tgt[:len(p_toks)] == [-100] * len(p_toks)
    # Transition position predicts first response token
    assert tgt[len(p_toks)] == r_toks[0]
    # Terminal target is EOS
    assert tgt[-1] == 1
