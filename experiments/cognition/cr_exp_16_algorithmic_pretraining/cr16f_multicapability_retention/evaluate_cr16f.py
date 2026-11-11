"""CR-EXP-16F Comprehensive Multi-Capability Evaluator
Evaluates all 12 factorial models (Arms A, B, C, D across seeds 2026, 3030, 4040)
across all mandated benchmark suites:
D1. CR-16 Copying Benchmark (100 probes)
D2. CR-16C Sealed Key-Value Retrieval (1,400 answerable + 100 unanswerable CAT9)
D3. Benchmark B1 (600 position-invariant probes) & B2 (150 permutation + 150 counterfactual pairs)
D4. Benchmark B3 Multi-Token Spans (500 probes, lengths 1..4)
D5. Benchmark B4 Calibrated Abstention (300 unanswerable probes)
D6. CR-16C CAT7 Unseen-Value Generalization (150 probes)
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

# Benchmark paths
CR16C_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"
B1_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b1_position_invariance_600.json"
B2_PERM_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b2_permutation_pairs_150.json"
B2_CF_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b2_counterfactual_pairs_150.json"
B3_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b3_multitoken_span_500.json"
B4_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b4_abstention_300.json"
BENCH_COPY_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"

LABELS_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study/answerability_labels.json"
with open(LABELS_FILE, encoding="utf-8") as f:
    ans_labels = json.load(f)

with open(CR16C_FILE, encoding="utf-8") as f: cr16c_bench = json.load(f)
with open(B1_FILE, encoding="utf-8") as f: b1_bench = json.load(f)
with open(B2_PERM_FILE, encoding="utf-8") as f: b2_perm_bench = json.load(f)
with open(B2_CF_FILE, encoding="utf-8") as f: b2_cf_bench = json.load(f)
with open(B3_FILE, encoding="utf-8") as f: b3_bench = json.load(f)
with open(B4_FILE, encoding="utf-8") as f: b4_bench = json.load(f)
with open(BENCH_COPY_FILE, encoding="utf-8") as f: b_copy_all = json.load(f)
copy_bench = [p for p in b_copy_all if p.get("category") == "exact_sequence_copying"]

# CR-16C CAT7 probes (unseen value)
cr16c_cat7_bench = [p for p in cr16c_bench if p.get("category") == "cat7_unseen_value"]

CR16F_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16f_multicapability_retention"
EVAL_DIR = CR16F_DIR / "eval_outputs"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def generate_greedy(model, prompt_str, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt_str)
    tokens = list(input_ids)
    
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            next_token = logits[0, -1, :].argmax().item()
            
        if next_token == 0:  # <pad> or eos
            break
        tokens.append(next_token)
        dec = tokenizer.decode([next_token])
        if "\n" in dec:
            break
            
    gen_tokens = tokens[len(input_ids):]
    gen_text = tokenizer.decode(gen_tokens).strip()
    return gen_text

def evaluate_single_model(ckpt_path, run_name):
    print(f"\n==========================================")
    print(f"Evaluating: {run_name}")
    print(f"Path: {ckpt_path}")
    print(f"==========================================")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"] if "config" in ckpt else ckpt["model_config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # D1. CR-16 Copying Benchmark (100)
    copy_results = []
    copy_exact = 0
    for p in copy_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        m = (pred == tgt)
        if m: copy_exact += 1
        copy_results.append({"id": p.get("id"), "prompt": p["prompt"], "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] CR-16 Copying: {copy_exact}/100")
    
    # D2. CR-16C Sealed Retrieval (1400 ans + 100 unans)
    cr16c_results = []
    c16_ans_corr = 0
    c16_ans_fa = 0
    c16_unans_rec = 0
    for p in cr16c_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        pid = p["id"]
        lbl = ans_labels["cr16c"][pid]["label"]
        m = (pred == tgt)
        if lbl == "ANSWERABLE":
            if m: c16_ans_corr += 1
            if pred == "[UNKNOWN]": c16_ans_fa += 1
        else:
            if pred == "[UNKNOWN]": c16_unans_rec += 1
        cr16c_results.append({"id": pid, "label": lbl, "category": p.get("category"), "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] CR-16C: Ans={c16_ans_corr}/1400 ({c16_ans_corr/1400:.1%}), False Abstain={c16_ans_fa}/1400, Unans Rec={c16_unans_rec}/100")
    
    # D6. CR-16C CAT7 Unseen-Value Generalization (150)
    cat7_results = []
    cat7_corr = 0
    for p in cr16c_cat7_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        m = (pred == tgt)
        if m: cat7_corr += 1
        cat7_results.append({"id": p["id"], "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] CR-16C CAT7 (Unseen-Value): {cat7_corr}/150 ({cat7_corr/150:.1%})")
    
    # D3. B1 Position Invariance (600)
    b1_results = []
    b1_corr = 0
    b1_fa = 0
    for p in b1_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        m = (pred == tgt)
        if m: b1_corr += 1
        if pred == "[UNKNOWN]": b1_fa += 1
        b1_results.append({"id": p["id"], "pos": p["target_pos"], "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] B1: Corr={b1_corr}/600 ({b1_corr/600:.1%}), False Abstain={b1_fa}/600")
    
    # D3b. B2 Permutation Pairs (150 pairs = 300 evaluations, checking pair consistency)
    b2_perm_results = []
    b2_perm_both_corr = 0
    for item in b2_perm_bench:
        p1 = item["probe_A"]["prompt"]
        t1 = item["expected"].strip()
        p2 = item["probe_B"]["prompt"]
        t2 = item["expected"].strip()
        pred1 = generate_greedy(model, p1)
        pred2 = generate_greedy(model, p2)
        m1 = (pred1 == t1)
        m2 = (pred2 == t2)
        both = m1 and m2
        if both: b2_perm_both_corr += 1
        b2_perm_results.append({"id": item["pair_id"], "pred1": pred1, "pred2": pred2, "both": both})
    print(f"[{run_name}] B2 Permutation: Both Correct={b2_perm_both_corr}/150 ({b2_perm_both_corr/150:.1%})")

    # D3c. B2 Counterfactual Pairs (150 pairs)
    b2_cf_results = []
    b2_cf_both_corr = 0
    for item in b2_cf_bench:
        p1 = item["probe_A"]["prompt"]
        t1 = item["probe_A"]["expected"].strip()
        p2 = item["probe_B"]["prompt"]
        t2 = item["probe_B"]["expected"].strip()
        pred1 = generate_greedy(model, p1)
        pred2 = generate_greedy(model, p2)
        m1 = (pred1 == t1)
        m2 = (pred2 == t2)
        both = m1 and m2
        if both: b2_cf_both_corr += 1
        b2_cf_results.append({"id": item["pair_id"], "pred1": pred1, "pred2": pred2, "both": both})
    print(f"[{run_name}] B2 Counterfactual: Both Correct={b2_cf_both_corr}/150 ({b2_cf_both_corr/150:.1%})")

    # D4. B3 Multi-Token Spans (500)
    b3_results = []
    b3_corr = 0
    b3_by_len = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in b3_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        l = p["target_len_tokens"]
        m = (pred == tgt)
        if m:
            b3_corr += 1
            b3_by_len[l] += 1
        b3_results.append({"id": p["id"], "target_len": l, "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] B3 Spans: Total={b3_corr}/500 ({b3_corr/500:.1%}), ByLen={b3_by_len}")
    
    # D5. B4 Calibrated Abstention (300)
    b4_results = []
    b4_rec = 0
    for p in b4_bench:
        pred = generate_greedy(model, p["prompt"])
        m = (pred == "[UNKNOWN]")
        if m: b4_rec += 1
        b4_results.append({"id": p["id"], "pred": pred, "match": m})
    print(f"[{run_name}] B4 Abstention: Recall={b4_rec}/300 ({b4_rec/300:.1%})")
    
    summary = {
        "run_name": run_name,
        "copying": {"exact": copy_exact, "total": 100, "rate": copy_exact / 100},
        "cr16c_answerable": {"correct": c16_ans_corr, "total": 1400, "rate": c16_ans_corr / 1400, "false_abstain": c16_ans_fa},
        "cr16c_unanswerable": {"recall": c16_unans_rec, "total": 100, "rate": c16_unans_rec / 100},
        "cr16c_cat7_unseen_val": {"correct": cat7_corr, "total": 150, "rate": cat7_corr / 150},
        "b1_position": {"correct": b1_corr, "total": 600, "rate": b1_corr / 600, "false_abstain": b1_fa},
        "b2_permutation": {"both_correct": b2_perm_both_corr, "total": 150, "rate": b2_perm_both_corr / 150},
        "b2_counterfactual": {"both_correct": b2_cf_both_corr, "total": 150, "rate": b2_cf_both_corr / 150},
        "b3_spans": {"total_correct": b3_corr, "total": 500, "rate": b3_corr / 500, "by_length": b3_by_len},
        "b4_abstention": {"recall": b4_rec, "total": 300, "rate": b4_rec / 300}
    }
    
    full_output = {
        "summary": summary,
        "predictions": {
            "copying": copy_results,
            "cr16c": cr16c_results,
            "cr16c_cat7": cat7_results,
            "b1": b1_results,
            "b2_permutation": b2_perm_results,
            "b2_counterfactual": b2_cf_results,
            "b3": b3_results,
            "b4": b4_results
        }
    }
    
    with open(EVAL_DIR / f"{run_name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    return summary

def main():
    arms = ["arm_a_replay25_lambda100", "arm_b_replay50_lambda100", "arm_c_replay25_lambda025", "arm_d_replay50_lambda025"]
    seeds = [2026, 3030, 4040]
    
    all_summaries = {}
    
    for arm in arms:
        for seed in seeds:
            run_name = f"{arm}_seed{seed}"
            ckpt_path = CR16F_DIR / "checkpoints" / run_name / f"{arm}_best.pt"
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found: {ckpt_path}")
                continue
            all_summaries[run_name] = evaluate_single_model(ckpt_path, run_name)
            
    summary_path = CR16F_DIR / "cr16f_evaluation_master_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll evaluations saved to {summary_path}")

if __name__ == "__main__":
    main()
