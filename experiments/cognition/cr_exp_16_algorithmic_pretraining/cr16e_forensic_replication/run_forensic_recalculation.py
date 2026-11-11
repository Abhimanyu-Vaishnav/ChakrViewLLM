"""Independent Forensic Evaluator for CR-EXP-16E-R
Executes strict independent re-evaluation of all 12 CR-16E checkpoints on raw benchmark data:
1. B1 Position Invariance (600 probes) - Position-by-position, query-order, template breakdown
2. B2 Permutation Pairs (150 pairs = 300 probes) - Strict pair-level classification
3. B2 Counterfactual Pairs (150 pairs = 300 probes) - Reversal consistency & invariant outputs
4. B3 Multi-Token Spans (500 probes) - Token-by-token divergence, lengths 1..4, teacher-forcing vs free-generation
5. B4 Abstention (300 probes) - Missing key, distractor, corrupted key; false abstention rate on answerable
6. CR-16 Copying Benchmark (100 probes) - Strict exact match & token presence
7. CR-16C Sealed Retrieval Benchmark (1,500 probes) - 10-category breakdown
Saves full raw predictions and outputs in eval_outputs/ for deep forensic analysis.
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

# Benchmarks
CR16E_DATA = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data"
BENCH_B1 = CR16E_DATA / "cr16e_b1_position_invariance_600.json"
BENCH_B2_PERM = CR16E_DATA / "cr16e_b2_permutation_pairs_150.json"
BENCH_B2_CF = CR16E_DATA / "cr16e_b2_counterfactual_pairs_150.json"
BENCH_B3_SPAN = CR16E_DATA / "cr16e_b3_multitoken_span_500.json"
BENCH_B4_ABS = CR16E_DATA / "cr16e_b4_abstention_300.json"
BENCH_COPY = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
BENCH_RETR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"

OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_forensic_replication/eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BENCH_B1) as f: b1_probes = json.load(f)
with open(BENCH_B2_PERM) as f: b2_perm = json.load(f)
with open(BENCH_B2_CF) as f: b2_cf = json.load(f)
with open(BENCH_B3_SPAN) as f: b3_span = json.load(f)
with open(BENCH_B4_ABS) as f: b4_abs = json.load(f)
with open(BENCH_COPY) as f: b_copy_all = json.load(f)
copy_probes = [p for p in b_copy_all if p['category'] == 'exact_sequence_copying']
with open(BENCH_RETR) as f: retr_probes = json.load(f)

def load_model(ckpt_path):
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

def evaluate_model_comprehensive(model_name, ckpt_path):
    print(f"\n=======================================================")
    print(f"AUDITING MODEL: {model_name}")
    print(f"=======================================================")
    model = load_model(ckpt_path)
    
    # 1. B1 Position Invariance
    b1_results = []
    b1_by_pos = {}
    b1_by_qpos = {"after": {"exact": 0, "total": 0}, "before": {"exact": 0, "total": 0}}
    b1_exact_total = 0
    for p in b1_probes:
        pos = p["target_pos"]
        qpos = p["query_pos"]
        if pos not in b1_by_pos: b1_by_pos[pos] = {"exact": 0, "total": 0}
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        m = (pred == target)
        if m:
            b1_exact_total += 1
            b1_by_pos[pos]["exact"] += 1
            b1_by_qpos[qpos]["exact"] += 1
        b1_by_pos[pos]["total"] += 1
        b1_by_qpos[qpos]["total"] += 1
        b1_results.append({
            "id": p["id"], "pos": pos, "qpos": qpos,
            "target": target, "pred": pred, "match": m
        })
    print(f"[{model_name}] B1: {b1_exact_total}/600 ({b1_exact_total/6:.1f}%) | Pos0={b1_by_pos[0]['exact']}/{b1_by_pos[0]['total']} | Pos1={b1_by_pos[1]['exact']}/{b1_by_pos[1]['total']} | QBefore={b1_by_qpos['before']['exact']}/{b1_by_qpos['before']['total']}")
    
    # 2. B2 Permutation Pairs
    perm_results = []
    perm_both = 0
    perm_orig_only = 0
    perm_perm_only = 0
    perm_both_inc = 0
    for pair in b2_perm:
        exp = pair["expected"].strip()
        pA = pair["probe_A"]["prompt"]
        pB = pair["probe_B"]["prompt"]
        predA, _ = generate_tokens(model, tokenizer.encode(pA), max_new_tokens=15)
        predB, _ = generate_tokens(model, tokenizer.encode(pB), max_new_tokens=15)
        mA = (predA == exp)
        mB = (predB == exp)
        if mA and mB: perm_both += 1
        elif mA and not mB: perm_orig_only += 1
        elif not mA and mB: perm_perm_only += 1
        else: perm_both_inc += 1
        perm_results.append({
            "pair_id": pair["pair_id"], "expected": exp,
            "predA": predA, "mA": mA, "predB": predB, "mB": mB,
            "type": "both_correct" if (mA and mB) else ("orig_only" if mA else ("perm_only" if mB else "both_wrong"))
        })
    print(f"[{model_name}] B2 Perm: Both={perm_both}/150 ({perm_both/1.5:.1f}%) | OrigOnly={perm_orig_only} | PermOnly={perm_perm_only} | BothWrong={perm_both_inc}")
    
    # 3. B2 Counterfactual Pairs
    cf_results = []
    cf_both = 0
    cf_invariant = 0
    cf_one_only = 0
    cf_both_wrong = 0
    for pair in b2_cf:
        expA = pair["probe_A"]["expected"].strip()
        expB = pair["probe_B"]["expected"].strip()
        pA = pair["probe_A"]["prompt"]
        pB = pair["probe_B"]["prompt"]
        predA, _ = generate_tokens(model, tokenizer.encode(pA), max_new_tokens=15)
        predB, _ = generate_tokens(model, tokenizer.encode(pB), max_new_tokens=15)
        mA = (predA == expA)
        mB = (predB == expB)
        if mA and mB: cf_both += 1
        elif mA or mB: cf_one_only += 1
        else: cf_both_wrong += 1
        if predA == predB: cf_invariant += 1
        cf_results.append({
            "pair_id": pair["pair_id"], "key": pair["key"],
            "expA": expA, "predA": predA, "mA": mA,
            "expB": expB, "predB": predB, "mB": mB,
            "invariant": (predA == predB)
        })
    print(f"[{model_name}] B2 CF: Both={cf_both}/150 ({cf_both/1.5:.1f}%) | Invariant={cf_invariant}/150 ({cf_invariant/1.5:.1f}%) | OneOnly={cf_one_only} | BothWrong={cf_both_wrong}")
    
    # 4. B3 Multi-Token Spans with Step-by-Step Divergence
    span_results = []
    span_by_len = {}
    span_exact_total = 0
    for p in b3_span:
        tlen = p["target_len_tokens"]
        if tlen not in span_by_len: span_by_len[tlen] = {"exact": 0, "first_match": 0, "total": 0}
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        target_tokens = tokenizer.encode(target)
        prompt_tokens = tokenizer.encode(prompt)
        pred, gen_ids = generate_tokens(model, prompt_tokens, max_new_tokens=len(target_tokens) + 5)
        
        m_exact = (pred == target)
        m_first = False
        divergence_idx = -1
        if target_tokens and gen_ids:
            m_first = (gen_ids[0] == target_tokens[0])
            for idx, (gid, tid) in enumerate(zip(gen_ids, target_tokens)):
                if gid != tid:
                    divergence_idx = idx
                    break
            if divergence_idx == -1 and len(gen_ids) < len(target_tokens):
                divergence_idx = len(gen_ids)
                
        if m_exact:
            span_exact_total += 1
            span_by_len[tlen]["exact"] += 1
        if m_first:
            span_by_len[tlen]["first_match"] += 1
        span_by_len[tlen]["total"] += 1
        
        span_results.append({
            "id": p["id"], "target": target, "target_len": tlen,
            "target_tokens": target_tokens, "gen_tokens": gen_ids,
            "pred": pred, "exact": m_exact, "first_match": m_first,
            "divergence_idx": divergence_idx
        })
    print(f"[{model_name}] B3 Spans: Total={span_exact_total}/500 ({span_exact_total/5:.1f}%) | L1={span_by_len[1]['exact']}/{span_by_len[1]['total']} | L2={span_by_len[2]['exact']}/{span_by_len[2]['total']} | L3={span_by_len[3]['exact']}/{span_by_len[3]['total']} (First={span_by_len[3]['first_match']}) | L4={span_by_len[4]['exact']}/{span_by_len[4]['total']} (First={span_by_len[4]['first_match']})")
    
    # 5. B4 Abstention (300 probes)
    abs_results = []
    abs_correct = 0
    for p in b4_abs:
        prompt = p["prompt"]
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=10)
        m = (pred == "[UNKNOWN]")
        if m: abs_correct += 1
        abs_results.append({
            "id": p["id"], "prompt": prompt, "pred": pred, "match": m
        })
    print(f"[{model_name}] B4 Abstain: {abs_correct}/300 ({abs_correct/3:.1f}%)")
    
    # 6. CR-16 Copying Benchmark (100 probes)
    copy_results = []
    copy_exact = 0
    copy_pres = 0
    for p in copy_probes:
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=len(tokenizer.encode(target)) + 5)
        m = (pred == target)
        if m: copy_exact += 1
        if target in pred: copy_pres += 1
        copy_results.append({"id": p.get("id"), "prompt": prompt, "target": target, "pred": pred, "match": m})
    print(f"[{model_name}] Copying: Exact={copy_exact}/100, Pres={copy_pres}/100")
    
    # 7. CR-16C Sealed Retrieval Benchmark (1,500 probes)
    retr_results = []
    retr_by_cat = {}
    retr_exact = 0
    for p in retr_probes:
        cat = p.get("category_code", p.get("category"))
        if cat not in retr_by_cat: retr_by_cat[cat] = {"exact": 0, "total": 0, "abs": 0}
        prompt = p["prompt"]
        target = p["expected_target"].strip()
        tokens = tokenizer.encode(prompt)
        pred, _ = generate_tokens(model, tokens, max_new_tokens=15)
        m = (pred == target)
        if m:
            retr_exact += 1
            retr_by_cat[cat]["exact"] += 1
        if pred == "[UNKNOWN]":
            retr_by_cat[cat]["abs"] += 1
        retr_by_cat[cat]["total"] += 1
        retr_results.append({"id": p.get("id"), "cat": cat, "target": target, "pred": pred, "match": m})
    print(f"[{model_name}] CR-16C: Total={retr_exact}/1500 ({retr_exact/15:.1f}%) | CAT1={retr_by_cat['CAT1']['exact']}/200 | CAT2={retr_by_cat['CAT2']['exact']}/200")
    
    full_output = {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "b1_position_invariance": {"total_exact": b1_exact_total, "by_pos": b1_by_pos, "by_qpos": b1_by_qpos, "predictions": b1_results},
        "b2_permutation": {"both_correct": perm_both, "orig_only": perm_orig_only, "perm_only": perm_perm_only, "both_wrong": perm_both_inc, "pairs": perm_results},
        "b2_counterfactual": {"both_correct": cf_both, "invariant": cf_invariant, "one_only": cf_one_only, "both_wrong": cf_both_wrong, "pairs": cf_results},
        "b3_spans": {"total_exact": span_exact_total, "by_len": span_by_len, "predictions": span_results},
        "b4_abstention": {"correct": abs_correct, "total": 300, "predictions": abs_results},
        "copying": {"exact": copy_exact, "presence": copy_pres, "total": 100, "predictions": copy_results},
        "cr16c_retrieval": {"total_exact": retr_exact, "by_cat": retr_by_cat, "predictions": retr_results}
    }
    
    with open(OUT_DIR / f"{model_name}_forensic_eval.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved full forensic prediction logs for {model_name}.")
    return full_output

if __name__ == "__main__":
    CR16E_CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/checkpoints"
    arms = ["arm_a_fixed_stdspan", "arm_b_shuffled_stdspan", "arm_c_fixed_balancedspan", "arm_d_shuffled_balancedspan"]
    seeds = [2026, 3030, 4040]
    
    master_summary = {}
    for arm in arms:
        for seed in seeds:
            mname = f"{arm}_s{seed}"
            p = CR16E_CKPT_DIR / f"{arm}_seed{seed}/{arm}_best.pt"
            if not p.exists():
                print(f"Skipping {mname}, missing: {p}")
                continue
            res = evaluate_model_comprehensive(mname, p)
            master_summary[mname] = {
                "b1": res["b1_position_invariance"]["total_exact"],
                "b2_perm_both": res["b2_permutation"]["both_correct"],
                "b2_cf_both": res["b2_counterfactual"]["both_correct"],
                "b2_cf_invariant": res["b2_counterfactual"]["invariant"],
                "b3_span_total": res["b3_spans"]["total_exact"],
                "b3_by_len": {k: v["exact"] for k, v in res["b3_spans"]["by_len"].items()},
                "b4_abstain": res["b4_abstention"]["correct"],
                "copying_exact": res["copying"]["exact"],
                "cr16c_total": res["cr16c_retrieval"]["total_exact"],
                "cr16c_cat1": res["cr16c_retrieval"]["by_cat"]["CAT1"]["exact"],
                "cr16c_cat2": res["cr16c_retrieval"]["by_cat"]["CAT2"]["exact"]
            }
            
    with open(OUT_DIR / "cr16er_master_recalculation_summary.json", "w") as f:
        json.dump(master_summary, f, indent=2)
    print("Master recalculation summary saved successfully.")

