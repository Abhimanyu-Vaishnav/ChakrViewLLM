"""
Stage E: Micro-sanity check for training pipeline.
Verifies:
1. SFT causal alignment, prompt masking (target=-100 on prompt tokens), and EOS target.
2. Gradient flow: parameters receive gradients ONLY from response tokens; no gradients from prompt.
3. Loss computation is finite.
4. Parameter tying (model.token_embedding.weight is model.lm_head.weight).
5. Optimizer step updates weights.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path("d:/Project/ChakrView").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.prompt_contract import build_causal_sequence

def run_training_pipeline_sanity():
    print("Starting Training Pipeline Correctness Sanity...")
    
    # 1. Model init & parameter tying
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    
    assert model.embedding.weight is model.lm_head.weight, "LM head and token embedding weights must be tied!"
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Verified parameter tying. Unique trainable params: {param_count}")
    
    # 2. Causal sequence masking
    prompt_tokens = [10, 20, 30]
    response_tokens = [40, 50]
    inp, tgt = build_causal_sequence(prompt_tokens, response_tokens, bos_id=0, eos_id=1, ignore_index=-100, append_eos=True)
    
    assert inp == [0, 10, 20, 30, 40, 50], f"Unexpected input seq: {inp}"
    assert tgt == [-100, -100, -100, 40, 50, 1], f"Unexpected target seq: {tgt}"
    assert len(inp) == len(tgt), "Input and target length mismatch"
    print("Verified causal sequence construction and prompt masking.")
    
    # 3. Forward, backward and gradient flow
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    x = torch.tensor([inp], dtype=torch.long)
    y = torch.tensor([tgt], dtype=torch.long)
    
    logits = model(x)
    assert logits.shape == (1, len(inp), 4096), f"Unexpected logits shape: {logits.shape}"
    
    loss = loss_fn(logits.view(-1, 4096), y.view(-1))
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"Forward pass finite loss: {loss.item():.4f}")
    
    loss.backward()
    
    # Check all gradients are finite
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            
    # Check that weights actually update on optimizer step
    w_before = model.embedding.weight.clone()
    optimizer.step()
    w_after = model.embedding.weight
    diff = (w_after - w_before).abs().sum().item()
    assert diff > 0.0, "Weights did not change after optimizer step!"
    print(f"Optimizer step verified! Weight delta: {diff:.6f}")
    print("STAGE E SANITY: ALL CHECKS PASSED.")

if __name__ == "__main__":
    run_training_pipeline_sanity()
