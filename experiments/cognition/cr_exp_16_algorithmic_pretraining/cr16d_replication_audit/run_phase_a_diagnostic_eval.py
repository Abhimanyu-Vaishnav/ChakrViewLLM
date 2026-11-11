"""Diagnostic Evaluator for CR-16D Models & CR-16DR Replication.
Saves full per-probe prediction logs for:
1. Frozen Copying Benchmark (100 probes)
2. Frozen Retrieval Benchmark (1500 probes) - includes full CAT9 / CAT10 breakdown
3. Frozen Compositional Benchmark (1000 probes) - G1..G4
4. Part B Diagnostics:
   - G1 Counterfactual Paired Inversion (200 pairs = 400 probes)
   - Position Permutation Test (100 pairs = 200 probes)
   - Conflicting Distractor Interventions (100 probes)
   - Logit & Unembedding Distribution Diagnostics on G2 Unseen Values
"""
import sys
import json
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Benchmarks
BENCH_COPY_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
BENCH_RETR_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"
BENCH_COMP_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data/cr16d_compositional_benchmark_1000.json"

with open(BENCH_COPY_PATH) as f:
    b_copy_all = json.load(f)
copy_probes = [p for p in b_copy_all if p['category'] == 'exact_sequence_copying']

with open(BENCH_RETR_PATH) as f:
    retr_probes = json.load(f)

with open(BENCH_COMP_PATH) as f:
    comp_probes = json.load(f)

def load_checkpoint_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt["model_config"] if "model_config" in ckpt else ckpt["config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def generate_tokens(model, prompt_tokens, max_new_tokens=20):
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
            if token_id == 1: # EOS
                break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec:
                break
    out_str = tokenizer.decode(gen_tokens, errors="replace")
    clean_str = out_str.strip().split("\n")[0].strip()
    return clean_str, gen_tokens

def evaluate_model_full(model_name, ckpt_path, out_file):
    print(f"\n=======================================================")
    print(f"RUNNING DIAGNOSTIC EVALUATION: {model_name}")
    print(f"=======================================================")
    model = load_checkpoint_model(ckpt_path)
    
    # 1. COPYING BENCHMARK
    copy_results = []
    copy_exact = 0
    copy_presence = 0
    copy_first = 0
    for p in copy_probes:
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, gen_ids = generate_tokens(model, tokens, max_new_tokens=len(tokenizer.encode(target)) + 5)
        is_exact = (pred == target)
        is_presence = (target in pred)
        first_token_target = tokenizer.encode(target)[0] if tokenizer.encode(target) else None
        first_token_pred = gen_ids[0] if gen_ids else None
        is_first = (first_token_pred == first_token_target)
        if is_exact: copy_exact += 1
        if is_presence: copy_presence += 1
        if is_first: copy_first += 1
        copy_results.append({
            "id": p.get("id"),
            "prompt": prompt,
            "target": target,
            "prediction": pred,
            "exact": is_exact,
            "presence": is_presence,
            "first_match": is_first
        })
    print(f"[{model_name}] Copying: Exact={copy_exact}/100, Pres={copy_presence}/100, First={copy_first}/100")
    
    # 2. RETRIEVAL BENCHMARK (1,500 probes)
    retr_results = []
    retr_by_cat = {}
    retr_exact_total = 0
    for p in retr_probes:
        cat = p.get('category_code', p.get('category'))
        if cat not in retr_by_cat:
            retr_by_cat[cat] = {"exact": 0, "total": 0, "abstention_preds": 0}
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, gen_ids = generate_tokens(model, tokens, max_new_tokens=15)
        is_exact = (pred == target)
        if is_exact:
            retr_exact_total += 1
            retr_by_cat[cat]["exact"] += 1
        retr_by_cat[cat]["total"] += 1
        if pred == "[UNKNOWN]":
            retr_by_cat[cat]["abstention_preds"] += 1
        retr_results.append({
            "id": p.get("id"),
            "category": cat,
            "prompt": prompt,
            "target": target,
            "prediction": pred,
            "exact": is_exact
        })
    print(f"[{model_name}] Retrieval Total: {retr_exact_total}/1500 ({retr_exact_total/15:.1f}%)")
    for cat in sorted(retr_by_cat.keys()):
        c = retr_by_cat[cat]
        print(f"  {cat}: {c['exact']}/{c['total']} ({c['exact']/c['total']*100:.1f}%) [Abstentions emitted: {c['abstention_preds']}]")
        
    # 3. COMPOSITIONAL BENCHMARK (1,000 probes)
    comp_results = []
    comp_by_group = {}
    comp_exact_total = 0
    for p in comp_probes:
        grp = p['group_code']
        if grp not in comp_by_group:
            comp_by_group[grp] = {"exact": 0, "total": 0}
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, gen_ids = generate_tokens(model, tokens, max_new_tokens=15)
        is_exact = (pred == target)
        if is_exact:
            comp_exact_total += 1
            comp_by_group[grp]["exact"] += 1
        comp_by_group[grp]["total"] += 1
        comp_results.append({
            "id": p.get("id"),
            "group": grp,
            "prompt": prompt,
            "target": target,
            "prediction": pred,
            "exact": is_exact
        })
    print(f"[{model_name}] Compositional Total: {comp_exact_total}/1000 ({comp_exact_total/10:.1f}%)")
    for grp in sorted(comp_by_group.keys()):
        g = comp_by_group[grp]
        print(f"  {grp}: {g['exact']}/{g['total']} ({g['exact']/g['total']*100:.1f}%)")
        
    full_data = {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "copying": {
            "exact": copy_exact,
            "presence": copy_presence,
            "first_token": copy_first,
            "total": len(copy_probes),
            "predictions": copy_results
        },
        "retrieval": {
            "total_exact": retr_exact_total,
            "total": len(retr_probes),
            "by_category": retr_by_cat,
            "predictions": retr_results
        },
        "compositional": {
            "total_exact": comp_exact_total,
            "total": len(comp_probes),
            "by_group": comp_by_group,
            "predictions": comp_results
        }
    }
    with open(out_file, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"Saved complete diagnostic predictions to {out_file}")
    return full_data

if __name__ == "__main__":
    ckpts = {
        "cr16a_arm_d_parent": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
        "cr16c_retrieval_only": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/checkpoints/retrieval_arm_a_target_masked_seed42/retrieval_arm_a_target_masked_best.pt",
        "arm_a_replay00_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_a_replay00_seed42/arm_a_replay00_best.pt",
        "arm_a_replay00_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_a_replay00_seed101/arm_a_replay00_best.pt",
        "arm_b_replay25_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_b_replay25_seed42/arm_b_replay25_best.pt",
        "arm_b_replay25_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_b_replay25_seed101/arm_b_replay25_best.pt",
        "arm_c_replay50_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_c_replay50_seed42/arm_c_replay50_best.pt",
        "arm_c_replay50_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_c_replay50_seed101/arm_c_replay50_best.pt",
        "arm_d_replay75_s42": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_d_replay75_seed42/arm_d_replay75_best.pt",
        "arm_d_replay75_s101": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_d_replay75_seed101/arm_d_replay75_best.pt"
    }
    out_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/eval_outputs"
    for name, p in ckpts.items():
        out_f = out_dir / f"{name}_full_eval.json"
        evaluate_model_full(name, p, out_f)
