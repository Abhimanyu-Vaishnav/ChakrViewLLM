"""Diagnostic Evaluator for Part B Interventions:
- Evaluates checkpoints on:
  1. G1 Counterfactual Paired Inversion (200 pairs = 400 probes)
  2. Position Permutation (100 pairs = 200 probes)
  3. Conflicting Distractors (100 probes)
  4. G2 Unseen Value Logit Diagnostic (loss, top-5 logits, rank)
"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/data"
with open(DATA_DIR / "cr16dr_counterfactual_pairs_200.json") as f:
    cf_pairs = json.load(f)
with open(DATA_DIR / "cr16dr_position_pairs_100.json") as f:
    pos_pairs = json.load(f)
with open(DATA_DIR / "cr16dr_distractor_probes_100.json") as f:
    dist_probes = json.load(f)

# G2 probes from compositional benchmark
BENCH_COMP_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data/cr16d_compositional_benchmark_1000.json"
with open(BENCH_COMP_PATH) as f:
    comp_all = json.load(f)
g2_probes = [p for p in comp_all if p['group_code'] == 'G2']

def load_checkpoint_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt["model_config"] if "model_config" in ckpt else ckpt["config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def generate_tokens(model, prompt_tokens, max_new_tokens=15):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    gen_tokens = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            gen_tokens.append(token_id)
            if token_id == 1:
                break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec:
                break
    out_str = tokenizer.decode(gen_tokens, errors="replace")
    return out_str.strip().split("\n")[0].strip()

def run_diagnostics_for_model(model_name, ckpt_path):
    print(f"\n=======================================================")
    print(f"RUNNING PART B DIAGNOSTICS: {model_name}")
    print(f"=======================================================")
    model = load_checkpoint_model(ckpt_path)
    
    # 1. COUNTERFACTUAL PAIRED INVERSION
    both_correct = 0
    a_only = 0
    b_only = 0
    both_incorrect = 0
    invariant_fail = 0
    
    cf_details = []
    for pair in cf_pairs:
        pA = pair["probe_A"]
        pB = pair["probe_B"]
        predA = generate_tokens(model, tokenizer.encode(pA["prompt"]))
        predB = generate_tokens(model, tokenizer.encode(pB["prompt"]))
        expA = pA["expected"]
        expB = pB["expected"]
        matchA = (predA == expA)
        matchB = (predB == expB)
        if matchA and matchB:
            both_correct += 1
        elif matchA and not matchB:
            a_only += 1
        elif not matchA and matchB:
            b_only += 1
        else:
            both_incorrect += 1
        if predA == predB:
            invariant_fail += 1
        cf_details.append({
            "pair_id": pair["pair_id"],
            "key": pair["key"],
            "predA": predA, "expA": expA, "matchA": matchA,
            "predB": predB, "expB": expB, "matchB": matchB,
            "invariant": (predA == predB)
        })
    print(f"[{model_name}] Counterfactual: Both={both_correct}/200 ({both_correct/2:.1f}%), A_only={a_only}, B_only={b_only}, Both_inc={both_incorrect}, Invariant={invariant_fail}")
    
    # 2. POSITION PERMUTATION
    pos_both_correct = 0
    pos_orig_only = 0
    pos_perm_only = 0
    pos_both_incorrect = 0
    pos_details = []
    for pair in pos_pairs:
        pred_orig = generate_tokens(model, tokenizer.encode(pair["orig_prompt"]))
        pred_perm = generate_tokens(model, tokenizer.encode(pair["perm_prompt"]))
        exp = pair["expected"]
        m_orig = (pred_orig == exp)
        m_perm = (pred_perm == exp)
        if m_orig and m_perm:
            pos_both_correct += 1
        elif m_orig and not m_perm:
            pos_orig_only += 1
        elif not m_orig and m_perm:
            pos_perm_only += 1
        else:
            pos_both_incorrect += 1
        pos_details.append({
            "pair_id": pair["pair_id"],
            "pred_orig": pred_orig, "m_orig": m_orig,
            "pred_perm": pred_perm, "m_perm": m_perm,
            "expected": exp
        })
    print(f"[{model_name}] Position Perm: Both={pos_both_correct}/100 ({pos_both_correct}%), Orig_only={pos_orig_only}, Perm_only={pos_perm_only}, Both_inc={pos_both_incorrect}")
    
    # 3. CONFLICTING DISTRACTORS
    dist_correct = 0
    dist_details = []
    for p in dist_probes:
        pred = generate_tokens(model, tokenizer.encode(p["prompt"]))
        exp = p["expected"]
        m = (pred == exp)
        if m: dist_correct += 1
        dist_details.append({
            "id": p["id"],
            "type": p["type"],
            "pred": pred,
            "expected": exp,
            "match": m
        })
    print(f"[{model_name}] Distractors: {dist_correct}/100 ({dist_correct}%)")
    
    # 4. G2 UNSEEN VALUE LOGIT DIAGNOSTICS (top 20 probes)
    g2_logit_details = []
    for p in g2_probes[:20]:
        prompt = p["prompt"]
        target = p["expected_target"]
        target_tokens = tokenizer.encode(target)
        if not target_tokens: continue
        first_target_id = target_tokens[0]
        
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            target_prob = probs[first_target_id].item()
            target_rank = (last_logits > last_logits[first_target_id]).sum().item() + 1
            top5_probs, top5_ids = torch.topk(probs, 5)
            top5_tokens = [tokenizer.decode([tid.item()]) for tid in top5_ids]
            
        pred = generate_tokens(model, prompt_tokens, max_new_tokens=10)
        g2_logit_details.append({
            "prompt": prompt,
            "target": target,
            "target_id": first_target_id,
            "target_prob": target_prob,
            "target_rank": target_rank,
            "pred": pred,
            "top5_predicted": list(zip(top5_tokens, [round(p, 4) for p in top5_probs.tolist()]))
        })
    print(f"[{model_name}] G2 Logit Diagnostics computed for 20 sample probes.")
    
    return {
        "counterfactual": {
            "both_correct": both_correct, "a_only": a_only, "b_only": b_only,
            "both_incorrect": both_incorrect, "invariant": invariant_fail,
            "pairs": cf_details
        },
        "position_permutation": {
            "both_correct": pos_both_correct, "orig_only": pos_orig_only,
            "perm_only": pos_perm_only, "both_incorrect": pos_both_incorrect,
            "pairs": pos_details
        },
        "distractors": {
            "correct": dist_correct, "total": len(dist_probes), "details": dist_details
        },
        "g2_unseen_value_logits": g2_logit_details
    }

if __name__ == "__main__":
    ckpts = {
        "arm_c_replay50_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_c_replay50_seed42/arm_c_replay50_best.pt",
        "arm_c_replay50_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_c_replay50_seed101/arm_c_replay50_best.pt",
        "arm_b_replay25_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_b_replay25_seed101/arm_b_replay25_best.pt",
        "arm_d_replay75_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_d_replay75_seed42/arm_d_replay75_best.pt"
    }
    out_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/eval_outputs"
    for name, p in ckpts.items():
        res = run_diagnostics_for_model(name, p)
        with open(out_dir / f"{name}_part_b_diagnostics.json", "w") as f:
            json.dump(res, f, indent=2)

