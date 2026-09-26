"""
Deterministic Parameter Accounting Verification Script for ChakrMicro v0.1.

Derives expected analytical counts from architectural dimensions:
V = 4096, d_model = 192, N = 6, H = 6, head_dim = 32, d_ff = 512, bias = False, tied = True.

Compares analytical numbers against PyTorch model parameter tensors.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def verify_parameter_accounting() -> bool:
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    
    # 1. Analytical Mathematical Derivation
    V = cfg.vocab_size          # 4096
    D = cfg.d_model             # 192
    N = cfg.n_layers            # 6
    D_ff = cfg.hidden_dim       # 512
    
    # Analytical counts:
    expected_emb = V * D                                # 4096 * 192 = 786,432
    expected_wq_per_layer = D * D                       # 192 * 192 = 36,864
    expected_wk_per_layer = D * D                       # 192 * 192 = 36,864
    expected_wv_per_layer = D * D                       # 192 * 192 = 36,864
    expected_wo_per_layer = D * D                       # 192 * 192 = 36,864
    expected_attn_per_layer = 4 * (D * D)               # 147,456
    expected_attn_total = N * expected_attn_per_layer   # 6 * 147,456 = 884,736
    
    expected_wgate_per_layer = D * D_ff                 # 192 * 512 = 98,304
    expected_wup_per_layer = D * D_ff                   # 192 * 512 = 98,304
    expected_wdown_per_layer = D_ff * D                 # 512 * 192 = 98,304
    expected_swiglu_per_layer = 3 * (D * D_ff)          # 294,912
    expected_swiglu_total = N * expected_swiglu_per_layer  # 6 * 294,912 = 1,769,472
    
    expected_norm_per_layer = 2 * D                     # 2 * 192 = 384
    expected_final_norm = D                             # 192
    expected_norm_total = (N * expected_norm_per_layer) + expected_final_norm  # 2,496
    
    expected_output_head = 0  # Reuses embedding weight matrix
    
    expected_total = (
        expected_emb +
        expected_attn_total +
        expected_swiglu_total +
        expected_norm_total +
        expected_output_head
    )  # 786,432 + 884,736 + 1,769,472 + 2,496 = 3,443,136
    
    # 2. Actual Code Parameter Accounting
    actual_emb = model.embedding.weight.numel()
    
    actual_wq = sum(l.attn.q_proj.weight.numel() for l in model.layers)
    actual_wk = sum(l.attn.k_proj.weight.numel() for l in model.layers)
    actual_wv = sum(l.attn.v_proj.weight.numel() for l in model.layers)
    actual_wo = sum(l.attn.out_proj.weight.numel() for l in model.layers)
    actual_attn = actual_wq + actual_wk + actual_wv + actual_wo
    
    actual_wgate = sum(l.ffn.gate_proj.weight.numel() for l in model.layers)
    actual_wup = sum(l.ffn.up_proj.weight.numel() for l in model.layers)
    actual_wdown = sum(l.ffn.down_proj.weight.numel() for l in model.layers)
    actual_swiglu = actual_wgate + actual_wup + actual_wdown
    
    actual_layer_norms = sum(l.norm_1.weight.numel() + l.norm_2.weight.numel() for l in model.layers)
    actual_final_norm = model.final_norm.weight.numel()
    actual_norm_total = actual_layer_norms + actual_final_norm
    
    # Unique parameter tensors in memory
    unique_params = set(model.parameters())
    actual_total = sum(p.numel() for p in unique_params)
    actual_trainable = sum(p.numel() for p in unique_params if p.requires_grad)
    
    # Verify weight tying sharing
    tied_sharing_verified = (model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr())
    
    print("=" * 80)
    print("CHAKRMICRO V0.1 - PARAMETER ACCOUNTING AUDIT")
    print("=" * 80)
    print(f"{'Component':<35} | {'Mathematical Expected':<22} | {'Code Measured':<15} | Status")
    print("-" * 80)
    
    checks = [
        ("Token Embedding (E)", expected_emb, actual_emb),
        ("Attention W_q (6 layers)", expected_wq_per_layer * N, actual_wq),
        ("Attention W_k (6 layers)", expected_wk_per_layer * N, actual_wk),
        ("Attention W_v (6 layers)", expected_wv_per_layer * N, actual_wv),
        ("Attention W_o (6 layers)", expected_wo_per_layer * N, actual_wo),
        ("Attention Total (6 layers)", expected_attn_total, actual_attn),
        ("SwiGLU W_gate (6 layers)", expected_wgate_per_layer * N, actual_wgate),
        ("SwiGLU W_up (6 layers)", expected_wup_per_layer * N, actual_wup),
        ("SwiGLU W_down (6 layers)", expected_wdown_per_layer * N, actual_wdown),
        ("SwiGLU Total (6 layers)", expected_swiglu_total, actual_swiglu),
        ("RMSNorm Layer Scales (6 layers)", expected_norm_per_layer * N, actual_layer_norms),
        ("Final RMSNorm Scale", expected_final_norm, actual_final_norm),
        ("Total Normalization Parameters", expected_norm_total, actual_norm_total),
        ("Tied LM Head Unique Params", expected_output_head, 0 if tied_sharing_verified else V * D),
        ("TOTAL UNIQUE PARAMETERS", expected_total, actual_total),
        ("TOTAL TRAINABLE PARAMETERS", expected_total, actual_trainable),
    ]
    
    all_matched = True
    for name, expected, actual in checks:
        matched = (expected == actual)
        if not matched:
            all_matched = False
        status_str = "MATCH [PASS]" if matched else "MISMATCH [FAIL]"
        print(f"{name:<35} | {expected:>22,} | {actual:>15,} | {status_str}")
        
    print("=" * 80)
    print(f"Weight Tying Physical Memory Sharing: {'VERIFIED [PASS]' if tied_sharing_verified else 'FAILED'}")
    print(f"Audit Overall Status: {'SUCCESS [PASS]' if all_matched and tied_sharing_verified else 'FAILURE [FAIL]'}")
    print("=" * 80)
    
    return all_matched and tied_sharing_verified


if __name__ == "__main__":
    success = verify_parameter_accounting()
    if not success:
        sys.exit(1)
