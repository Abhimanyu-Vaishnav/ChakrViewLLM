"""Grand Comprehensive Evaluator for CR-EXP-16E Factorial Models:
Evaluates across all 12 factorial checkpoints (Arms A, B, C, D across Seeds 2026, 3030, 4040):
1. CR-16 Original Frozen Copying Benchmark (100 probes) -> Gate 4
2. CR-16C Original Frozen Retrieval Benchmark (1,500 probes) -> Gate 5
3. CR-16E Sealed Benchmark B1: Position-Invariance Probes (600 probes) -> Gate 1
4. CR-16E Sealed Benchmark B2: Matched Permutation Pairs (150 pairs = 300 probes) & Counterfactual Pairs (150 pairs = 300 probes) -> Gate 1 & Gate 2
5. CR-16E Sealed Benchmark B3: Multi-Token Span Benchmark (500 probes across lengths 1, 2, 3, 4) -> Gate 3
6. CR-16E Sealed Benchmark B4: Calibrated Abstention Benchmark (300 probes) -> Gate 6
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
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data"
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BENCH_COPY_PATH) as f:
    b_copy_all = json.load(f)
copy_probes = [p for p in b_copy_all if p['category'] == 'exact_sequence_copying']

with open(BENCH_RETR_PATH) as f:
    retr_probes = json.load(f)

with open(DATA_DIR / "cr16e_b1_position_invariance_600.json") as f:
    b1_probes = json.load(f)
with open(DATA_DIR / "cr16e_b2_permutation_pairs_150.json") as f:
    b2_perm_pairs = json.load(f)
with open(DATA_DIR / "cr16e_b2_counterfactual_pairs_150.json") as f:
    b2_cf_pairs = json.load(f)
with open(DATA_DIR / "cr16e_b3_multitoken_span_500.json") as f:
    b3_span_probes = json.load(f)
with open(DATA_DIR / "cr16e_b4_abstention_300.json") as f:
    b4_abs_probes = json.load(f)

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

def evaluate_cr16e_model(model_name, ckpt_path):
    print(f"\n=======================================================")
    print(f"EVALUATING CR-16E MODEL: {model_name}")
    print(f"=======================================================")
    model = load_checkpoint_model(ckpt_path)
    
    # 1. COPYING BENCHMARK (100 probes)
    copy_exact = 0
    copy_pres = 0
    for p in copy_probes:
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=len(tokenizer.encode(target)) + 5)
        if pred == target: copy_exact += 1
        if target in pred: copy_pres += 1
    print(f"[{model_name}] Copying: Exact={copy_exact}/100, Pres={copy_pres}/100")
    
    # 2. RETRIEVAL BENCHMARK (1,500 probes)
    retr_exact = 0
    retr_by_cat = {}
    for p in retr_probes:
        cat = p.get('category_code', p.get('category'))
        if cat not in retr_by_cat:
            retr_by_cat[cat] = {"exact": 0, "total": 0}
        prompt = p['prompt']
        target = p['expected_target'].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        if pred == target:
            retr_exact += 1
            retr_by_cat[cat]["exact"] += 1
        retr_by_cat[cat]["total"] += 1
    print(f"[{model_name}] CR-16C Retr: Total={retr_exact}/1500 ({retr_exact/15:.1f}%) | CAT1={retr_by_cat['CAT1']['exact']}/200 | CAT2={retr_by_cat['CAT2']['exact']}/200")
    
    # 3. B1: POSITION INVARIANCE (600 probes)
    b1_exact = 0
    b1_by_pos = {}
    b1_by_qpos = {"after": {"exact": 0, "total": 0}, "before": {"exact": 0, "total": 0}}
    for p in b1_probes:
        pos = p["target_pos"]
        qpos = p["query_pos"]
        if pos not in b1_by_pos: b1_by_pos[pos] = {"exact": 0, "total": 0}
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        if pred == target:
            b1_exact += 1
            b1_by_pos[pos]["exact"] += 1
            b1_by_qpos[qpos]["exact"] += 1
        b1_by_pos[pos]["total"] += 1
        b1_by_qpos[qpos]["total"] += 1
    print(f"[{model_name}] B1 Pos Invariance: Total={b1_exact}/600 ({b1_exact/6:.1f}%)")
    for pos in sorted(b1_by_pos.keys()):
        bp = b1_by_pos[pos]
        print(f"   Pos {pos}: {bp['exact']}/{bp['total']} ({bp['exact']/bp['total']*100:.1f}%)")
    print(f"   Query Before Facts: {b1_by_qpos['before']['exact']}/{b1_by_qpos['before']['total']} ({b1_by_qpos['before']['exact']/max(b1_by_qpos['before']['total'], 1)*100:.1f}%)")
    
    # 4. B2: POSITION PERMUTATION (150 pairs) & COUNTERFACTUAL (150 pairs)
    perm_both_correct = 0
    for pair in b2_perm_pairs:
        pA = pair["probe_A"]["prompt"]
        pB = pair["probe_B"]["prompt"]
        exp = pair["expected"].strip()
        predA, _ = generate_tokens(model, tokenizer.encode(pA), max_new_tokens=15)
        predB, _ = generate_tokens(model, tokenizer.encode(pB), max_new_tokens=15)
        if predA == exp and predB == exp:
            perm_both_correct += 1
    print(f"[{model_name}] B2 Permutation Pairs: Both={perm_both_correct}/150 ({perm_both_correct/1.5:.1f}%)")
    
    cf_both_correct = 0
    cf_invariant_fail = 0
    for pair in b2_cf_pairs:
        pA = pair["probe_A"]["prompt"]
        pB = pair["probe_B"]["prompt"]
        expA = pair["probe_A"]["expected"].strip()
        expB = pair["probe_B"]["expected"].strip()
        predA, _ = generate_tokens(model, tokenizer.encode(pA), max_new_tokens=15)
        predB, _ = generate_tokens(model, tokenizer.encode(pB), max_new_tokens=15)
        if predA == expA and predB == expB:
            cf_both_correct += 1
        if predA == predB:
            cf_invariant_fail += 1
    print(f"[{model_name}] B2 Counterfactual Pairs: Both={cf_both_correct}/150 ({cf_both_correct/1.5:.1f}%), Invariant={cf_invariant_fail}/150")
    
    # 5. B3: MULTI-TOKEN SPAN BENCHMARK (500 probes)
    span_exact = 0
    span_by_len = {}
    for p in b3_span_probes:
        tlen = p["target_len_tokens"]
        if tlen not in span_by_len: span_by_len[tlen] = {"exact": 0, "total": 0}
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=20)
        if pred == target:
            span_exact += 1
            span_by_len[tlen]["exact"] += 1
        span_by_len[tlen]["total"] += 1
    print(f"[{model_name}] B3 Spans: Total={span_exact}/500 ({span_exact/5:.1f}%)")
    for tl in sorted(span_by_len.keys()):
        sb = span_by_len[tl]
        print(f"   Len {tl}: {sb['exact']}/{sb['total']} ({sb['exact']/sb['total']*100:.1f}%)")
        
    # 6. B4: CALIBRATED ABSTENTION BENCHMARK (300 probes)
    abs_correct = 0
    for p in b4_abs_probes:
        prompt = p["prompt"]
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=10)
        if pred == "[UNKNOWN]":
            abs_correct += 1
    print(f"[{model_name}] B4 Abstention: {abs_correct}/300 ({abs_correct/3:.1f}%)")
    
    return {
        "copying": {"exact": copy_exact, "presence": copy_pres, "total": 100},
        "retrieval_cr16c": {"total_exact": retr_exact, "by_cat": retr_by_cat},
        "b1_position_invariance": {"total_exact": b1_exact, "by_pos": b1_by_pos, "by_qpos": b1_by_qpos},
        "b2_permutation": {"both_correct": perm_both_correct, "total": 150},
        "b2_counterfactual": {"both_correct": cf_both_correct, "invariant": cf_invariant_fail, "total": 150},
        "b3_spans": {"total_exact": span_exact, "by_len": span_by_len},
        "b4_abstention": {"correct": abs_correct, "total": 300}
    }

if __name__ == "__main__":
    ckpt_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/checkpoints"
    arms = ["arm_a_fixed_stdspan", "arm_b_shuffled_stdspan", "arm_c_fixed_balancedspan", "arm_d_shuffled_balancedspan"]
    seeds = [2026, 3030, 4040]
    
    grand_results = {}
    for arm in arms:
        for seed in seeds:
            mname = f"{arm}_s{seed}"
            p = ckpt_dir / f"{arm}_seed{seed}/{arm}_best.pt"
            if not p.exists():
                print(f"Skipping {mname}, checkpoint does not exist: {p}")
                continue
            res = evaluate_cr16e_model(mname, p)
            grand_results[mname] = res
            
    with open(OUT_DIR / "cr16e_grand_evaluation_summary.json", "w") as f:
        json.dump(grand_results, f, indent=2)
    print("Saved CR-16E grand evaluation summary successfully.")

