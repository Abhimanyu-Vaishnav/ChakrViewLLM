"""Grand Evaluator for Replication Checkpoints (Seeds 202, 303, 404).
Evaluates all 12 replication models on:
1. Frozen CR-16 Copying Benchmark (100 probes)
2. Frozen CR-16C Retrieval Benchmark (1,500 probes)
3. Frozen CR-16D Compositional Generalization Benchmark (1,000 probes)
Saves complete grand scorecard and raw outputs.
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
            if token_id == 1:
                break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec:
                break
    out_str = tokenizer.decode(gen_tokens, errors="replace")
    clean_str = out_str.strip().split("\n")[0].strip()
    return clean_str, gen_tokens

def evaluate_replication_model(model_name, ckpt_path):
    print(f"\n=======================================================")
    print(f"EVALUATING REPLICATION MODEL: {model_name}")
    print(f"=======================================================")
    model = load_checkpoint_model(ckpt_path)
    
    # 1. COPYING
    copy_exact = 0
    copy_pres = 0
    copy_first = 0
    for p in copy_probes:
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, gen_ids = generate_tokens(model, tokens, max_new_tokens=len(tokenizer.encode(target)) + 5)
        if pred == target: copy_exact += 1
        if target in pred: copy_pres += 1
        first_token_target = tokenizer.encode(target)[0] if tokenizer.encode(target) else None
        first_token_pred = gen_ids[0] if gen_ids else None
        if first_token_pred == first_token_target: copy_first += 1
        
    print(f"[{model_name}] Copying: Exact={copy_exact}/100, Pres={copy_pres}/100, First={copy_first}/100")
    
    # 2. RETRIEVAL
    retr_exact = 0
    retr_by_cat = {}
    for p in retr_probes:
        cat = p.get('category_code', p.get('category'))
        if cat not in retr_by_cat:
            retr_by_cat[cat] = {"exact": 0, "total": 0, "abstentions": 0}
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        if pred == target:
            retr_exact += 1
            retr_by_cat[cat]["exact"] += 1
        retr_by_cat[cat]["total"] += 1
        if pred == "[UNKNOWN]":
            retr_by_cat[cat]["abstentions"] += 1
            
    print(f"[{model_name}] Retrieval: Overall={retr_exact}/1500 ({retr_exact/15:.1f}%) | CAT1={retr_by_cat['CAT1']['exact']}/200 | CAT2={retr_by_cat['CAT2']['exact']}/200 | CAT9={retr_by_cat['CAT9']['exact']}/100")
    
    # 3. COMPOSITIONAL
    comp_exact = 0
    comp_by_grp = {}
    for p in comp_probes:
        grp = p['group_code']
        if grp not in comp_by_grp:
            comp_by_grp[grp] = {"exact": 0, "total": 0}
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        if pred == target:
            comp_exact += 1
            comp_by_grp[grp]["exact"] += 1
        comp_by_grp[grp]["total"] += 1
        
    print(f"[{model_name}] Compositional: Total={comp_exact}/1000 ({comp_exact/10:.1f}%) | G1={comp_by_grp['G1']['exact']}/250 | G2={comp_by_grp['G2']['exact']}/250 | G3={comp_by_grp['G3']['exact']}/250 | G4={comp_by_grp['G4']['exact']}/250")
    
    return {
        "copying": {"exact": copy_exact, "presence": copy_pres, "first_token": copy_first, "total": 100},
        "retrieval": {"total_exact": retr_exact, "total": 1500, "by_cat": retr_by_cat},
        "compositional": {"total_exact": comp_exact, "total": 1000, "by_group": comp_by_grp}
    }

if __name__ == "__main__":
    rep_ckpt_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/checkpoints"
    out_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/eval_outputs"
    
    arms = ["arm_a_replay00", "arm_b_replay25", "arm_c_replay50", "arm_d_replay75"]
    seeds = [202, 303, 404]
    
    results = {}
    for arm in arms:
        for seed in seeds:
            model_name = f"{arm}_s{seed}"
            ckpt_path = rep_ckpt_dir / f"{arm}_seed{seed}/{arm}_best.pt"
            if not ckpt_path.exists():
                print(f"Skipping {model_name}, checkpoint not found: {ckpt_path}")
                continue
            res = evaluate_replication_model(model_name, ckpt_path)
            results[model_name] = res
            
    with open(out_dir / "cr16dr_seeds_202_303_404_eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved replication evaluation summary successfully.")

