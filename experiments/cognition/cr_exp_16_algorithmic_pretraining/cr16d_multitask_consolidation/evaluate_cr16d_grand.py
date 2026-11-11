"""CR-EXP-16D Grand Multi-Task Evaluation Harness
Evaluates all consolidation models across:
1. Frozen CR-16 Sequence Copying Benchmark (100 probes) -> Checks catastrophic forgetting / retention.
2. Frozen CR-16C Key-Value Retrieval Benchmark (1,500 probes) -> Checks retrieval acquisition.
3. New CR-16D Compositional Generalization Benchmark (1,000 probes) -> Checks G1..G4 generalization.
"""
import sys
import json
import torch
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data"
CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints"
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Benchmarks
BENCH_COPY_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
BENCH_RETR_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"
BENCH_COMP_PATH = DATA_DIR / "cr16d_compositional_benchmark_1000.json"

with open(BENCH_COPY_PATH) as f: b_copy_all = json.load(f)
copy_probes = [p for p in b_copy_all if p['category'] == 'exact_sequence_copying'] # 100 probes

with open(BENCH_RETR_PATH) as f: retr_probes = json.load(f) # 1500 probes
with open(BENCH_COMP_PATH) as f: comp_probes = json.load(f) # 1000 probes

print(f"Loaded benchmarks: {len(copy_probes)} copy probes, {len(retr_probes)} retr probes, {len(comp_probes)} comp probes.")

# Models to evaluate
models = {
    # Baselines
    "cr16a_arm_d_parent": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
    "cr16c_retrieval_only": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/checkpoints/retrieval_arm_a_target_masked_seed42/retrieval_arm_a_target_masked_best.pt",
    # CR-16D Consolidation Models Seed 42
    "arm_a_replay00_s42": CKPT_DIR / "arm_a_replay00_seed42/arm_a_replay00_best.pt",
    "arm_b_replay25_s42": CKPT_DIR / "arm_b_replay25_seed42/arm_b_replay25_best.pt",
    "arm_c_replay50_s42": CKPT_DIR / "arm_c_replay50_seed42/arm_c_replay50_best.pt",
    "arm_d_replay75_s42": CKPT_DIR / "arm_d_replay75_seed42/arm_d_replay75_best.pt",
    # CR-16D Consolidation Models Seed 101
    "arm_a_replay00_s101": CKPT_DIR / "arm_a_replay00_seed101/arm_a_replay00_best.pt",
    "arm_b_replay25_s101": CKPT_DIR / "arm_b_replay25_seed101/arm_b_replay25_best.pt",
    "arm_c_replay50_s101": CKPT_DIR / "arm_c_replay50_seed101/arm_c_replay50_best.pt",
    "arm_d_replay75_s101": CKPT_DIR / "arm_d_replay75_seed101/arm_d_replay75_best.pt",
}

def generate(model, prompt_tokens, max_new_tokens=20):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if token_id == 1: break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec: break
    out_str = tokenizer.decode(input_ids[0, len(prompt_tokens):].tolist(), errors="replace")
    return out_str.strip().split("\n")[0].strip()

grand_summary = {}

for model_name, ckpt_path in models.items():
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"=======================================================")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt["model_config"] if "model_config" in ckpt else ckpt["config"]
    
    model = ChakrMicro(config).to(device)
    model.load_state_dict(state)
    model.eval()
    
    # 1. EVALUATE COPYING RETENTION (100 probes)
    copy_exact = 0
    copy_presence = 0
    copy_first = 0
    for p in copy_probes:
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred = generate(model, tokens)
        
        # first token
        with torch.no_grad():
            out = model(torch.tensor([tokens], device=device))
            logits = out[0] if isinstance(out, tuple) else out
            f_tok = logits[0, -1, :].argmax().item()
            tgt_tokens = tokenizer.encode(target)
            if len(tgt_tokens) > 0 and f_tok == tgt_tokens[0]:
                copy_first += 1
                
        if pred == target:
            copy_exact += 1
        t_words = set(target.split())
        p_words = set(pred.split())
        if len(t_words) > 0 and t_words.issubset(p_words):
            copy_presence += 1
            
    print(f"[{model_name}] Copying Retention: Exact = {copy_exact}/100 ({copy_exact}%) | Presence = {copy_presence}/100 | FirstToken = {copy_first}/100")
    
    # 2. EVALUATE RETRIEVAL ACQUISITION (1,500 probes)
    retr_cat_exact = {}
    retr_exact = 0
    for p in retr_probes:
        c = p['category_code']
        retr_cat_exact.setdefault(c, {'exact': 0, 'total': 0})
        retr_cat_exact[c]['total'] += 1
        prompt = p['prompt']
        target = p['expected_target'].strip()
        pred = generate(model, tokenizer.encode(prompt))
        if pred.lower() == target.lower():
            retr_cat_exact[c]['exact'] += 1
            retr_exact += 1
            
    cat1_acc = retr_cat_exact['CAT1']['exact'] / retr_cat_exact['CAT1']['total']
    cat2_acc = retr_cat_exact['CAT2']['exact'] / retr_cat_exact['CAT2']['total']
    print(f"[{model_name}] Retrieval: Overall = {retr_exact}/1500 ({retr_exact/15:.1f}%) | Direct(CAT1) = {cat1_acc*100:.1f}% | Reassign(CAT2) = {cat2_acc*100:.1f}%")
    
    # 3. EVALUATE COMPOSITIONAL GENERALIZATION (1,000 probes)
    comp_group_exact = {}
    comp_exact = 0
    for p in comp_probes:
        g = p['group_code']
        comp_group_exact.setdefault(g, {'exact': 0, 'total': 0})
        comp_group_exact[g]['total'] += 1
        prompt = p['prompt']
        target = p['expected_target'].strip()
        pred = generate(model, tokenizer.encode(prompt))
        if pred.lower() == target.lower():
            comp_group_exact[g]['exact'] += 1
            comp_exact += 1
            
    g1_acc = comp_group_exact['G1']['exact'] / comp_group_exact['G1']['total']
    g2_acc = comp_group_exact['G2']['exact'] / comp_group_exact['G2']['total']
    g3_acc = comp_group_exact['G3']['exact'] / comp_group_exact['G3']['total']
    g4_acc = comp_group_exact['G4']['exact'] / comp_group_exact['G4']['total']
    print(f"[{model_name}] Compositional G1..G4: Total = {comp_exact}/1000 ({comp_exact/10:.1f}%) | G1(Combos)={g1_acc*100:.1f}% | G2(UnseenVal)={g2_acc*100:.1f}% | G3(UnseenKey)={g3_acc*100:.1f}% | G4(Both)={g4_acc*100:.1f}%")
    
    grand_summary[model_name] = {
        "copying": {
            "exact": copy_exact,
            "presence": copy_presence,
            "first_token": copy_first
        },
        "retrieval": {
            "total_exact": retr_exact,
            "by_category": retr_cat_exact
        },
        "compositional": {
            "total_exact": comp_exact,
            "by_group": comp_group_exact
        }
    }

with open(OUT_DIR / "cr16d_grand_evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(grand_summary, f, indent=2)

print("\nGrand Evaluation Complete! Output saved to eval_outputs/cr16d_grand_evaluation_summary.json")

