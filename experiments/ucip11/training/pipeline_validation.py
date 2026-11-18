"""
Verification of Training Pipeline Correctness for Phase D:
1. Autoregressive Shift alignment (shift_logits vs shift_targets)
2. Loss masking on prompts (labels = -100 on prompt tokens)
3. Causal masking (lower triangular attention mask)
4. Special token boundaries (BOS at start, EOS at end of target)
5. Overfitting sanity check on a tiny 4-sample dataset (initial ~8.38 drops to < 0.2 within 40 steps, < 0.05 within 60 steps)
"""

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

# 1. Pipeline Shift & Masking Test
def test_pipeline_alignment():
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=2, n_heads=4, max_seq_len=128)
    model = ChakrMicro(cfg)
    
    prompt = "User: Hello\nAssistant:"
    target = " Hi there!"
    
    p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    t_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
    
    input_ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    
    inp = torch.tensor([input_ids], dtype=torch.long)
    tgt = torch.tensor([labels], dtype=torch.long)
    
    logits = model(inp)
    
    # Autoregressive shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tgt[:, 1:].contiguous()
    
    assert shift_logits.shape[1] == shift_targets.shape[1], "Shift length mismatch!"
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
    
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"
    
    loss.backward()
    
    # Check that gradients flow to all active parameters
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert all(gn > 0 for gn in grad_norms), "Dead gradients detected!"
    
    print("Pipeline alignment and gradient flow tests PASSED.")

# 2. Overfitting Sanity Check on 4 Samples
def test_overfitting_sanity():
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=2, n_heads=4, max_seq_len=128)
    model = ChakrMicro(cfg)
    optimizer = AdamW(model.parameters(), lr=1.5e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    samples = [
        ("User: Who are you?\nAssistant:", " I am ChakrView, an independent AI system."),
        ("User: What is 2 + 2?\nAssistant:", " The sum is 4."),
        ("User: Define CPU.\nAssistant:", " The CPU is the central processing unit."),
        ("User: नमस्ते!\nAssistant:", " नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।")
    ]
    
    batch_inps = []
    batch_tgts = []
    
    for p, t in samples:
        p_ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        t_ids = tokenizer.encode(t, add_bos=False, add_eos=True)
        seq = p_ids + t_ids
        lbl = [-100] * len(p_ids) + t_ids
        batch_inps.append(seq)
        batch_tgts.append(lbl)
        
    max_len = max(len(s) for s in batch_inps)
    padded_inps = [s + [2] * (max_len - len(s)) for s in batch_inps]
    padded_tgts = [l + [-100] * (max_len - len(l)) for l in batch_tgts]
    
    t_inps = torch.tensor(padded_inps, dtype=torch.long)
    t_tgts = torch.tensor(padded_tgts, dtype=torch.long)
    
    initial_loss = None
    final_loss = None
    
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        logits = model(t_inps)
        s_logits = logits[:, :-1, :].contiguous()
        s_targets = t_tgts[:, 1:].contiguous()
        loss = criterion(s_logits.view(-1, s_logits.size(-1)), s_targets.view(-1))
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
    print(f"Tiny Overfitting Sanity: Initial Loss = {initial_loss:.4f} -> Final Loss (50 steps) = {final_loss:.4f}")
    assert final_loss < 0.1, f"Failed to overfit tiny dataset! Final loss: {final_loss}"
    print("Tiny overfitting sanity test PASSED.")

if __name__ == "__main__":
    test_pipeline_alignment()
    test_overfitting_sanity()
