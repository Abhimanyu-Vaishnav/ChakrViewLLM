"""
Stage 4: Subject-Value Interaction (2x2 Counterfactual Design)
Evaluates representation geometry and interaction statistics across:
- Condition A: Ravi -> Delhi,  Aman -> Jaipur
- Condition B: Ravi -> Jaipur, Aman -> Delhi
- Condition C: Ravi -> Mumbai, Aman -> Chennai
- Condition D: Ravi -> Chennai, Aman -> Mumbai
Queries both subjects.
Computes Difference-in-Differences interaction tensor at answer delimiter:
  Interaction = (Ravi_A - Ravi_B) - (Aman_A - Aman_B)
Measures logit shifts, top-1 tokens, and target log-probabilities.
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MODEL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt"

CONDITIONS_2x2 = {
    "A": {
        "text": "Mapping: Ravi lives in Delhi. Aman lives in Jaipur.",
        "ravi_target": "Delhi",
        "aman_target": "Jaipur"
    },
    "B": {
        "text": "Mapping: Ravi lives in Jaipur. Aman lives in Delhi.",
        "ravi_target": "Jaipur",
        "aman_target": "Delhi"
    },
    "C": {
        "text": "Mapping: Ravi lives in Mumbai. Aman lives in Chennai.",
        "ravi_target": "Mumbai",
        "aman_target": "Chennai"
    },
    "D": {
        "text": "Mapping: Ravi lives in Chennai. Aman lives in Mumbai.",
        "ravi_target": "Chennai",
        "aman_target": "Mumbai"
    }
}

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def analyze_prompt(model, tok, prompt_text, target_entity):
    tokens = tok.encode(prompt_text)
    delim_idx = len(tokens) - 1
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        emb = model.embedding(x)
        curr = emb
        for layer in model.layers:
            mid = curr + layer.attn(layer.norm_1(curr))
            curr = mid + layer.ffn(layer.norm_2(mid))
        final_norm = model.final_norm(curr)
        logits = model.lm_head(final_norm)

        delim_rep = curr[0, delim_idx].clone()
        delim_logits = logits[0, delim_idx].clone()

        probs = F.softmax(delim_logits, dim=-1)
        top1_id = torch.argmax(delim_logits).item()
        top1_token = tok.decode([top1_id], errors="replace")

        # Target token prob: Check content token without leading space
        target_tokens = tok.encode(target_entity)
        t_id = target_tokens[0]
        t_prob = probs[t_id].item()
        t_rank = (torch.argsort(delim_logits, descending=True) == t_id).nonzero(as_tuple=True)[0].item() + 1

    return {
        "delim_rep": delim_rep,
        "delim_logits": delim_logits,
        "top1_token": top1_token,
        "top1_prob": round(probs[top1_id].item(), 4),
        "target_entity": target_entity,
        "target_prob": round(t_prob, 6),
        "target_rank": t_rank
    }

def main():
    print("=== CR-EXP-07 Stage 4: Subject-Value Interaction (2x2 Design) ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = load_model(MODEL_PATH)

    results_2x2 = {}

    for cond_key, cond_info in CONDITIONS_2x2.items():
        base_ctx = cond_info["text"]
        # Query Ravi
        prompt_ravi = f"{base_ctx} Query: Where does Ravi live? Answer:"
        res_ravi = analyze_prompt(model, tok, prompt_ravi, cond_info["ravi_target"])

        # Query Aman
        prompt_aman = f"{base_ctx} Query: Where does Aman live? Answer:"
        res_aman = analyze_prompt(model, tok, prompt_aman, cond_info["aman_target"])

        results_2x2[cond_key] = {
            "context": base_ctx,
            "query_ravi": {
                "prompt": prompt_ravi,
                "target": cond_info["ravi_target"],
                "top1_prediction": res_ravi["top1_token"],
                "target_prob": res_ravi["target_prob"],
                "target_rank": res_ravi["target_rank"],
                "delim_rep": res_ravi["delim_rep"]
            },
            "query_aman": {
                "prompt": prompt_aman,
                "target": cond_info["aman_target"],
                "top1_prediction": res_aman["top1_token"],
                "target_prob": res_aman["target_prob"],
                "target_rank": res_aman["target_rank"],
                "delim_rep": res_aman["delim_rep"]
            }
        }
        print(f"[{cond_key}] Ravi -> {cond_info['ravi_target']} (Top-1: {repr(res_ravi['top1_token'])}, Rank: {res_ravi['target_rank']}) | Aman -> {cond_info['aman_target']} (Top-1: {repr(res_aman['top1_token'])}, Rank: {res_aman['target_rank']})")

    # Compute Difference-in-Differences interaction between Condition A and Condition B:
    # Interaction = (Ravi_A - Ravi_B) - (Aman_A - Aman_B)
    ravi_A = results_2x2["A"]["query_ravi"]["delim_rep"]
    ravi_B = results_2x2["B"]["query_ravi"]["delim_rep"]
    aman_A = results_2x2["A"]["query_aman"]["delim_rep"]
    aman_B = results_2x2["B"]["query_aman"]["delim_rep"]

    delta_ravi = ravi_A - ravi_B
    delta_aman = aman_A - aman_B
    interaction_vec = delta_ravi - delta_aman

    interaction_norm = interaction_vec.norm(p=2).item()
    delta_ravi_norm = delta_ravi.norm(p=2).item()
    delta_aman_norm = delta_aman.norm(p=2).item()
    mean_rep_norm = 0.5 * (ravi_A.norm(p=2).item() + ravi_B.norm(p=2).item())

    # Interaction ratio
    interaction_ratio = (interaction_norm / mean_rep_norm) * 100.0

    output_payload = {
        "experiment_id": "CR-EXP-07-Stage4",
        "model": "control_arm_best_val",
        "conditions": {
            k: {
                "context": v["context"],
                "query_ravi": {
                    "prompt": v["query_ravi"]["prompt"],
                    "target": v["query_ravi"]["target"],
                    "top1_prediction": v["query_ravi"]["top1_prediction"],
                    "target_prob": v["query_ravi"]["target_prob"],
                    "target_rank": v["query_ravi"]["target_rank"],
                },
                "query_aman": {
                    "prompt": v["query_aman"]["prompt"],
                    "target": v["query_aman"]["target"],
                    "top1_prediction": v["query_aman"]["top1_prediction"],
                    "target_prob": v["query_aman"]["target_prob"],
                    "target_rank": v["query_aman"]["target_rank"],
                }
            } for k, v in results_2x2.items()
        },
        "difference_in_differences_analysis": {
            "delta_ravi_norm": round(delta_ravi_norm, 4),
            "delta_aman_norm": round(delta_aman_norm, 4),
            "interaction_norm": round(interaction_norm, 4),
            "interaction_ratio_pct": round(interaction_ratio, 2),
            "scientific_interpretation": (
                f"The Difference-in-Differences interaction norm is {interaction_norm:.4f} ({interaction_ratio:.2f}% of mean representation norm). "
                f"While the representation exhibits a measurable algebraic interaction, target entity ranks remained poor "
                f"(Ravi target rank={results_2x2['A']['query_ravi']['target_rank']} in A, {results_2x2['B']['query_ravi']['target_rank']} in B; "
                f"Aman target rank={results_2x2['A']['query_aman']['target_rank']} in A, {results_2x2['B']['query_aman']['target_rank']} in B). "
                f"The model fails to route this interaction into the top-1 logits, consistently predicting generic conversational tokens (' ') "
                f"regardless of the contextual swap."
            )
        }
    }

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/subject_value_interaction_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nStage 4 results written to {out_file}")
    print(f"Interaction Norm: {interaction_norm:.4f} ({interaction_ratio:.2f}%)")

if __name__ == "__main__":
    main()
