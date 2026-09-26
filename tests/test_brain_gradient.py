"""
Gradient and learnability smoke test for ChakrMicro (Phase 8).

Verifies:
- Loss is finite (no NaN / Inf).
- Gradients exist for all trainable parameters.
- Parameters actually change under SGD/AdamW update steps.
- Loss monotonically or consistently decreases on a tiny synthetic sequence ("ABABAB...").
"""

import torch
import torch.nn.functional as F
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def test_gradient_flow_and_parameter_updates():
    """
    Run 15 optimization steps on a repeating synthetic sequence:
    Pattern: [10, 20, 10, 20, 10, 20, 10, 20]
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.train()
    
    # Repeating token pattern
    tokens = torch.tensor([[10, 20, 10, 20, 10, 20, 10, 20]], dtype=torch.long)
    input_ids = tokens[:, :-1]   # [1, 7]: [10, 20, 10, 20, 10, 20, 10]
    target_ids = tokens[:, 1:]   # [1, 7]: [20, 10, 20, 10, 20, 10, 20]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    
    # Store initial parameter snapshot for comparison
    initial_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    for step in range(15):
        optimizer.zero_grad()
        logits = model(input_ids)  # [1, 7, 4096]
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target_ids.view(-1))
        
        assert not torch.isnan(loss), f"Loss became NaN at step {step}"
        assert not torch.isinf(loss), f"Loss became Inf at step {step}"
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        # Verify gradients exist and are finite for all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for parameter {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                
        optimizer.step()
        final_loss = loss.item()
        
    # 1. Loss must decrease from initial loss
    assert final_loss < initial_loss, (
        f"Smoke training failed to decrease loss: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
    
    # 2. Parameters must actually change
    changed_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            diff = torch.norm(param.detach() - initial_params[name]).item()
            if diff > 1e-6:
                changed_params += 1
                
    assert changed_params > 0, "No model parameters changed during optimization steps!"
