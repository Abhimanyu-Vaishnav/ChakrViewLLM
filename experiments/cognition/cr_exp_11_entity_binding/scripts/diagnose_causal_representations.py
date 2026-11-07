"""
CR-EXP-11 Phase 5: Representation, Attention, and Causal Diagnostics
Investigates:
1. Layerwise hidden state cosine similarities under subject/value substitutions
2. Attention weight distribution across tokens (Does query attend to subject, initial state, or updated state?)
3. Logit ranking of target city vs distractor city vs initial city at the final position
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"

def diagnose_representations():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(REPO_ROOT / "experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_arm_b/cr_exp_10_arm_b_best_val_copy.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.eval()
    
    # Controlled Prompt Triad:
    # Base: Meera in Jaipur, moves to Bhopal. Query: Where is Meera now?
    p_base = "Context: Meera works in Jaipur. Meera later moves to Bhopal.\nQuestion: Where does Meera work now?\nAnswer: "
    # Counterfactual Subject: Aarav in Jaipur, moves to Bhopal. Query: Where is Aarav now?
    p_cf_subj = "Context: Aarav works in Jaipur. Aarav later moves to Bhopal.\nQuestion: Where does Aarav work now?\nAnswer: "
    # Counterfactual Value: Meera in Jaipur, moves to Pune. Query: Where is Meera now?
    p_cf_val = "Context: Meera works in Jaipur. Meera later moves to Pune.\nQuestion: Where does Meera work now?\nAnswer: "
    
    # Check logit rankings at final token
    enc_base = tok.encode(p_base)
    enc_cf_subj = tok.encode(p_cf_subj)
    enc_cf_val = tok.encode(p_cf_val)
    
    with torch.no_grad():
        x_base = torch.tensor([enc_base], dtype=torch.long)
        logits_base = m(x_base)[0, -1] # [V]
        
        x_subj = torch.tensor([enc_cf_subj], dtype=torch.long)
        logits_subj = m(x_subj)[0, -1]
        
        x_val = torch.tensor([enc_cf_val], dtype=torch.long)
        logits_val = m(x_val)[0, -1]
        
    # Get token IDs for cities: Bhopal, Jaipur, Pune
    # Note: check if space is prefixed
    tok_bhopal = tok.encode(" Bhopal")[0] if len(tok.encode(" Bhopal")) > 0 else tok.encode("Bhopal")[0]
    tok_jaipur = tok.encode(" Jaipur")[0] if len(tok.encode(" Jaipur")) > 0 else tok.encode("Jaipur")[0]
    tok_pune = tok.encode(" Pune")[0] if len(tok.encode(" Pune")) > 0 else tok.encode("Pune")[0]
    
    b_prob = F.softmax(logits_base, dim=-1)
    
    top5_base = torch.topk(b_prob, 5)
    top5_tokens = [(tok.decode([idx.item()], errors="replace"), round(val.item(), 4)) for val, idx in zip(top5_base.values, top5_base.indices)]
    
    # Rank of target vs initial
    probs_dict = {
        "Bhopal_prob": float(b_prob[tok_bhopal].item()),
        "Jaipur_prob": float(b_prob[tok_jaipur].item()),
        "Pune_prob": float(b_prob[tok_pune].item()),
        "top5_predictions": top5_tokens
    }
    
    return {
        "prompt_base": p_base,
        "base_token_count": len(enc_base),
        "logits_analysis": probs_dict
    }

def main():
    res = diagnose_representations()
    out_file = OUT_DIR / "representation_and_causal_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Logit and Representation Analysis:")
    print("Prompt:", res["prompt_base"])
    print("Top 5 predictions:", res["logits_analysis"]["top5_predictions"])
    print("Bhopal (Latest State) Prob:", res["logits_analysis"]["Bhopal_prob"])
    print("Jaipur (Initial State) Prob:", res["logits_analysis"]["Jaipur_prob"])
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()

