import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.mixture_sft_engine import MixtureConfig, UCIP15Engine

cfg = MixtureConfig(
    output_dir=Path("experiments/ucip17/scratch_grad"),
    base_checkpoint=Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
    steps=10,
    sft_batch_size=8,
    replay_batch_size=2,
    seed=101
)
engine = UCIP15Engine(cfg)
rng = np.random.RandomState(101)

cos_sims = []
gnorm_sft_raw = []
gnorm_rep_raw = []
gnorm_sft_weighted = []
gnorm_rep_weighted = []
gnorm_comb = []
update_sft_share = []

# Weights: lambda_sft = 0.85, lambda_rep = 0.15
lambda_sft = 0.85
lambda_rep = 0.15

for step in range(5):
    x_sft, y_sft, sft_tokens = engine.sft_train.get_batch(batch_size=8, rng=rng)
    x_rep, y_rep, rep_tokens = engine.replay_train.get_batch(batch_size=2, rng=rng)
    
    # 1. SFT forward & gradient
    engine.optimizer.zero_grad()
    logits_sft = engine.model(x_sft)
    l_sft = engine.loss_fn_sft(logits_sft.reshape(-1, 4096), y_sft.reshape(-1)) / sft_tokens
    l_sft.backward()
    grad_sft = [p.grad.clone() for p in engine.model.parameters() if p.grad is not None]
    norm_sft = torch.sqrt(sum((g**2).sum() for g in grad_sft)).item()
    gnorm_sft_raw.append(norm_sft)
    gnorm_sft_weighted.append(lambda_sft * norm_sft)
    
    # 2. Replay forward & gradient
    engine.optimizer.zero_grad()
    logits_rep = engine.model(x_rep)
    l_rep = engine.loss_fn_replay(logits_rep.reshape(-1, 4096), y_rep.reshape(-1)) / rep_tokens
    l_rep.backward()
    grad_rep = [p.grad.clone() for p in engine.model.parameters() if p.grad is not None]
    norm_rep = torch.sqrt(sum((g**2).sum() for g in grad_rep)).item()
    gnorm_rep_raw.append(norm_rep)
    gnorm_rep_weighted.append(lambda_rep * norm_rep)
    
    # Cosine similarity between SFT and Replay gradients
    dot_prod = sum((g_s * g_r).sum() for g_s, g_r in zip(grad_sft, grad_rep)).item()
    cos_sim = dot_prod / (norm_sft * norm_rep + 1e-12)
    cos_sims.append(cos_sim)
    
    # Combined gradient: G = lambda_sft * grad_sft + lambda_rep * grad_rep
    grad_comb = [lambda_sft * g_s + lambda_rep * g_r for g_s, g_r in zip(grad_sft, grad_rep)]
    norm_comb = torch.sqrt(sum((g**2).sum() for g in grad_comb)).item()
    gnorm_comb.append(norm_comb)
    
    # Actual projection / share of update aligned with SFT:
    # dot(grad_comb, grad_sft) / norm_comb^2
    proj_sft = sum((g_c * g_s).sum() for g_c, g_s in zip(grad_comb, grad_sft)).item()
    update_sft_share.append((lambda_sft * norm_sft**2 + lambda_sft * lambda_rep * dot_prod) / (norm_comb**2 + 1e-12))

print("=== Empirical Gradient Decomposition (L = 0.85*L_sft + 0.15*L_rep) ===")
print(f"Mean Raw SFT Gradient Norm      : {np.mean(gnorm_sft_raw):.4f}")
print(f"Mean Raw Replay Gradient Norm   : {np.mean(gnorm_rep_raw):.4f}")
print(f"Mean Weighted SFT Norm (0.85*N) : {np.mean(gnorm_sft_weighted):.4f}")
print(f"Mean Weighted Replay Norm(0.15*N): {np.mean(gnorm_rep_weighted):.4f}")
print(f"Gradient Cosine Similarity      : {np.mean(cos_sims):.4f}")
print(f"Combined Gradient Norm          : {np.mean(gnorm_comb):.4f}")
print(f"Effective SFT Update Share      : {np.mean(update_sft_share)*100:.2f}%")
print(f"Norm Ratio (Weighted SFT / Sum) : {np.mean(gnorm_sft_weighted)/(np.mean(gnorm_sft_weighted)+np.mean(gnorm_rep_weighted))*100:.2f}%")
