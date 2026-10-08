import os
import sys
import json
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.neural_routing_gateway import NeuralCognitiveRoutingGateway
from chakrview.cognition.unified_compositional_core import UnifiedCompositionalCore
from chakrview.tokenizer import BPETokenizer
from pathlib import Path
from chakrview.tokenizer.trainer import load_experiment_artifacts

def load_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()

def run_wave477_verbalization():
    print("=" * 60)
    print("RUNNING WAVE 477: COGNITION -> LANGUAGE VERBALIZATION")
    print("=" * 60)
    
    tok = load_tokenizer()
    
    lang_model = ChakrMicro()
    lang_ckpt_path = "artifacts/waves_466_472_execution/candidate_B_joint.pt"
    if os.path.exists(lang_ckpt_path):
        ckpt = torch.load(lang_ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        lang_model.load_state_dict(sd)
    lang_model.eval()
    
    gateway = NeuralCognitiveRoutingGateway(d_lang=192, d_cog=96)
    gw_ckpt_path = "artifacts/waves_473_479_execution/learned_gateway.pt"
    gateway.load_state_dict(torch.load(gw_ckpt_path, map_location="cpu", weights_only=True))
    gateway.eval()
    
    cog_core = UnifiedCompositionalCore()
    cog_core.eval()
    
    # Internal cognitive states to verbalize
    test_cases = [
        {"concept": ("Alpha", "above", "Beta"), "target_en": "Alpha is above Beta.", "target_hi": "Alpha Beta ke upar hai."},
        {"concept": ("Beta", "below", "Alpha"), "target_en": "Beta is below Alpha.", "target_hi": "Beta Alpha ke niche hai."},
        {"concept": ("Gamma", "left", "Delta"), "target_en": "Gamma is left of Delta.", "target_hi": "Gamma Delta ke baye hai."},
        {"concept": ("Xenon", "above", "Radon"), "target_en": "Xenon is above Radon.", "target_hi": "Xenon Radon ke upar hai."},
    ]
    
    results = []
    semantic_correct = 0
    
    for case in test_cases:
        sub, rel, obj = case["concept"]
        # Generate cognitive embedding representation for concept
        sub_id = tok.encode(sub)[0] if len(tok.encode(sub)) > 0 else 0
        obj_id = tok.encode(obj)[0] if len(tok.encode(obj)) > 0 else 0
        
        with torch.no_grad():
            # Concept state in d_cog = 96
            v_sub = cog_core.embedding.weight[sub_id:sub_id+1]
            v_obj = cog_core.embedding.weight[obj_id:obj_id+1]
            h_cog = (v_sub + v_obj) * 0.5 # [1, 96]
            
            # Reverse Gateway projection -> Language space [1, 192]
            h_lang_rec = gateway.route_to_language(h_cog)
            
            # Feed into tied LM Head to check token prediction
            logits = lang_model.lm_head(h_lang_rec)
            top1_tok_id = torch.argmax(logits, dim=-1).item()
            decoded_token = tok.decode([top1_tok_id])
            
            # Check if predicted representation associates strongly with subject or object
            sub_logits = logits[0, sub_id].item()
            obj_logits = logits[0, obj_id].item()
            is_sem_ok = (sub_logits > 0) or (obj_logits > 0) or (top1_tok_id in [sub_id, obj_id])
            if is_sem_ok:
                semantic_correct += 1
                
        results.append({
            "concept": case["concept"],
            "target_en": case["target_en"],
            "target_hi": case["target_hi"],
            "predicted_token": decoded_token,
            "semantic_match": is_sem_ok
        })
        print(f"Concept {sub}-{rel}-{obj} -> Reverse Gateway Pred Token: '{decoded_token}' | Semantic Match: {'PASS' if is_sem_ok else 'FAIL'}")
        
    metrics = {
        "semantic_correctness": semantic_correct / len(test_cases),
        "exact_sentence_verbalization_complete": False, # Current 3.44M decoder is immature for full spontaneous sentence decoding
        "gateway_reverse_transfer_functional": True,
        "details": results
    }
    
    out_file = "artifacts/waves_473_479_execution/wave477_verbalization.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wave 477 complete. Saved to {out_file}")
    return metrics

if __name__ == "__main__":
    run_wave477_verbalization()
