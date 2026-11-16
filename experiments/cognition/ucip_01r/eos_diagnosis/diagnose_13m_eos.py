import os
import sys
import json
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from scripts.run_m3_evaluations import load_tokenizer_artifacts

def diagnose_eos():
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    ckpt_path = REPO_ROOT / "experiments/cognition/ucip_01/training/ucip01_candidate_best.pt"
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    dummy_cfg = ModelConfig()
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(dummy_cfg, k)})
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    test_prompts = [
        "User: Hello assistant, who are you and how can you help me?\nAssistant:",
        "User: What is the capital of France?\nAssistant:",
        "User: Calculate 5 + 7.\nAssistant:",
        "User: नमस्ते! आप कौन हैं?\nAssistant:",
        "User: Mera naam Priya hai. Aapka naam kya hai?\nAssistant:",
        "[Memory: Name: Priya, City: Pune]\nUser: What is my name and where do I live?\nAssistant:",
        "User: My project is Yatrajivi.\nAssistant: Got it.\nUser: I renamed it to ChakrView.\nAssistant: Renamed.\nUser: What is my project called now?\nAssistant:",
        "User: What did I eat for dinner three weeks ago?\nAssistant:"
    ]
    
    results = []
    
    for p in test_prompts:
        input_ids = tok.encode(p, add_bos=True, add_eos=False)
        t_in = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(t_in)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            eos_prob = probs[1].item()
            eos_logit = logits[1].item()
            
            # Rank of EOS (token 1)
            sorted_indices = torch.argsort(logits, descending=True)
            eos_rank = (sorted_indices == 1).nonzero(as_tuple=True)[0].item() + 1
            
            top5_indices = sorted_indices[:5].tolist()
            top5_tokens = [tok.decode([idx]) for idx in top5_indices]
            top5_probs = [probs[idx].item() for idx in top5_indices]
            
        results.append({
            "prompt": p,
            "input_tokens_count": len(input_ids),
            "eos_token_id": 1,
            "eos_probability": eos_prob,
            "eos_logit": eos_logit,
            "eos_rank": eos_rank,
            "top5_indices": top5_indices,
            "top5_tokens": top5_tokens,
            "top5_probabilities": top5_probs
        })
        
        print(f"Prompt: {repr(p[:40])}...")
        print(f"  EOS Prob: {eos_prob:.4f} (Rank {eos_rank})")
        print(f"  Top 1 Token: {repr(top5_tokens[0])} (Prob: {top5_probs[0]:.4f})")
        print()
        
    out_dir = REPO_ROOT / "experiments/cognition/ucip_01r/eos_diagnosis"
    with open(out_dir / "eos_diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Diagnostics written to {out_dir / 'eos_diagnostic_results.json'}")

if __name__ == "__main__":
    diagnose_eos()
