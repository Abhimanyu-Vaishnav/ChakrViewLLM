import os
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.unified_compositional_core import UnifiedCompositionalCore
from chakrview.cognition.neural_routing_gateway import NeuralCognitiveRoutingGateway
from chakrview.tokenizer import BPETokenizer
from pathlib import Path
from chakrview.tokenizer.trainer import load_experiment_artifacts

def load_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()

def run_wave474_gateway_acquisition():
    print("=" * 60)
    print("RUNNING WAVE 474: LEARNED GATEWAY ACQUISITION")
    print("=" * 60)
    
    os.makedirs("artifacts/waves_473_479_execution", exist_ok=True)
    
    # 1. Instantiate Frozen Language Model & Relational Core
    lang_model = ChakrMicro()
    lang_ckpt_path = "artifacts/waves_466_472_execution/candidate_B_joint.pt"
    if os.path.exists(lang_ckpt_path):
        ckpt = torch.load(lang_ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        lang_model.load_state_dict(sd)
    lang_model.eval()
    for p in lang_model.parameters():
        p.requires_grad = False
        
    cog_core = UnifiedCompositionalCore()
    cog_core.eval()
    for p in cog_core.parameters():
        p.requires_grad = False
        
    gateway = NeuralCognitiveRoutingGateway(d_lang=192, d_cog=96)
    gateway.train()
    
    # 2. Synthetic Language-Relational Pairs
    entities = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    relations = ["above", "below", "left", "right", "north", "south"]
    
    tok = load_tokenizer()
    
    # Target dataset: Learn to map language prompt representation to cognitive core target space
    data_pairs = []
    for _ in range(64):
        e1, e2 = random.sample(entities, 2)
        rel = random.choice(relations)
        text = f"{e1} is {rel} of {e2}."
        data_pairs.append((text, e1, rel, e2))
        
    optimizer = optim.AdamW(gateway.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    t0 = time.perf_counter()
    initial_loss = None
    final_loss = None
    
    for epoch in range(15):
        epoch_loss = 0.0
        for text, e1, rel, e2 in data_pairs:
            input_ids = torch.tensor([tok.encode(text)], dtype=torch.long)
            if input_ids.shape[1] == 0:
                continue
            
            with torch.no_grad():
                # Extract language hidden state at last token
                h_lang = lang_model.embedding(input_ids)
                for layer in lang_model.layers:
                    h_lang = layer(h_lang)
                h_lang = lang_model.final_norm(h_lang) # [1, T, 192]
                
                # Cognitive reference embedding
                cog_ref = cog_core.embedding(input_ids) # [1, T, 96]
                
            # Forward gateway projection
            pred_cog = gateway.route_to_cognition(h_lang)
            loss_fwd = criterion(pred_cog, cog_ref)
            
            # Reverse gateway projection
            rec_lang = gateway.route_to_language(pred_cog)
            loss_rev = criterion(rec_lang, h_lang)
            
            loss = loss_fwd + loss_rev
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(data_pairs)
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss
        
    t_elapsed = time.perf_counter() - t0
    print(f"Gateway Training: Initial Loss={initial_loss:.4f} -> Final Loss={final_loss:.4f} in {t_elapsed:.2f}s")
    
    # Save checkpoint
    ckpt_path = "artifacts/waves_473_479_execution/learned_gateway.pt"
    torch.save(gateway.state_dict(), ckpt_path)
    
    metrics = {
        "gateway_params": gateway.total_param_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": (initial_loss - final_loss) / initial_loss,
        "training_time_s": t_elapsed,
        "trainable": True,
        "baseline_delta_w": 0,
        "candidate_checkpoint": ckpt_path
    }
    
    out_file = "artifacts/waves_473_479_execution/wave474_gateway_acquisition.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wave 474 complete. Saved to {out_file}")
    return metrics

if __name__ == "__main__":
    run_wave474_gateway_acquisition()
