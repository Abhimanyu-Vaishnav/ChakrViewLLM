"""CR-16E-R2 Comprehensive Evaluator for G1 vs G2 Controlled Study
Evaluates all 6 checkpoints across:
1. CR-16 Copying Benchmark (100 probes)
2. CR-16C Sealed Retrieval (1,400 Answerable Probes)
3. CR-16C Sealed Retrieval (100 Unanswerable Probes - CAT9)
4. Benchmark B1 Position Invariance (600 probes)
5. Benchmark B3 Multi-Token Spans (500 probes)
6. Benchmark B4 Calibrated Abstention (300 probes)
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
CR16C_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"
B1_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b1_position_invariance_600.json"
B3_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b3_multitoken_span_500.json"
B4_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_b4_abstention_300.json"
BENCH_COPY_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"

LABELS_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study/answerability_labels.json"
with open(LABELS_FILE, encoding="utf-8") as f:
    ans_labels = json.load(f)

with open(CR16C_FILE, encoding="utf-8") as f: cr16c_bench = json.load(f)
with open(B1_FILE, encoding="utf-8") as f: b1_bench = json.load(f)
with open(B3_FILE, encoding="utf-8") as f: b3_bench = json.load(f)
with open(B4_FILE, encoding="utf-8") as f: b4_bench = json.load(f)
with open(BENCH_COPY_FILE, encoding="utf-8") as f: b_copy_all = json.load(f)
copy_bench = [p for p in b_copy_all if p.get("category") == "exact_sequence_copying"]

OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study"
EVAL_DIR = OUT_DIR / "eval_outputs"
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

def evaluate_model(ckpt_path, run_name):
    print(f"\nEvaluating: {run_name} from {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"] if "config" in ckpt else ckpt["model_config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # 1. CR-16 Copying
    copy_results = []
    copy_exact = 0
    for p in copy_bench:
        pred = generate_greedy(model, p["prompt"])
        tgt = p["expected_target"].strip()
        m = (pred == tgt)
        if m: copy_exact += 1
        copy_results.append({"id": p.get("id"), "prompt": p["prompt"], "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] Copying: {copy_exact}/100")
    
    # 2. CR-16C Retrieval (split into 1400 ans and 100 unans)
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
        else: # UNANSWERABLE
            if pred == "[UNKNOWN]": c16_unans_rec += 1
        cr16c_results.append({"id": pid, "label": lbl, "target": tgt, "pred": pred, "match": m})
    print(f"[{run_name}] CR-16C: Ans={c16_ans_corr}/1400, False Abstain={c16_ans_fa}/1400, Unans Rec={c16_unans_rec}/100")
    
    # 3. B1 Position Invariance (600)
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
    print(f"[{run_name}] B1: Corr={b1_corr}/600, False Abstain={b1_fa}/600")
    
    # 4. B3 Multi-token Spans (500)
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
    print(f"[{run_name}] B3: Total={b3_corr}/500, ByLen={b3_by_len}")
    
    # 5. B4 Calibrated Abstention (300)
    b4_results = []
    b4_rec = 0
    for p in b4_bench:
        pred = generate_greedy(model, p["prompt"])
        m = (pred == "[UNKNOWN]")
        if m: b4_rec += 1
        b4_results.append({"id": p["id"], "pred": pred, "match": m})
    print(f"[{run_name}] B4: Recall={b4_rec}/300")
    
    summary = {
        "run_name": run_name,
        "copying": {"exact": copy_exact, "total": 100},
        "cr16c_answerable": {"correct": c16_ans_corr, "total": 1400, "false_abstain": c16_ans_fa},
        "cr16c_unanswerable": {"recall": c16_unans_rec, "total": 100},
        "b1_position": {"correct": b1_corr, "total": 600, "false_abstain": b1_fa},
        "b3_spans": {"total_correct": b3_corr, "total": 500, "by_length": b3_by_len},
        "b4_abstention": {"recall": b4_rec, "total": 300}
    }
    
    full_output = {
        "summary": summary,
        "predictions": {
            "copying": copy_results,
            "cr16c": cr16c_results,
            "b1": b1_results,
            "b3": b3_results,
            "b4": b4_results
        }
    }
    
    with open(EVAL_DIR / f"{run_name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    return summary

if __name__ == "__main__":
    runs = [
        ("arm_g1_enabled_seed2026", "arm_g1_enabled_seed2026/arm_g1_enabled_best.pt"),
        ("arm_g1_enabled_seed3030", "arm_g1_enabled_seed3030/arm_g1_enabled_best.pt"),
        ("arm_g1_enabled_seed4040", "arm_g1_enabled_seed4040/arm_g1_enabled_best.pt"),
        ("arm_g2_masked_seed2026", "arm_g2_masked_seed2026/arm_g2_masked_best.pt"),
        ("arm_g2_masked_seed3030", "arm_g2_masked_seed3030/arm_g2_masked_best.pt"),
        ("arm_g2_masked_seed4040", "arm_g2_masked_seed4040/arm_g2_masked_best.pt"),
    ]
    all_summaries = {}
    for rname, rel_path in runs:
        ckpt_path = OUT_DIR / "checkpoints" / rel_path
        all_summaries[rname] = evaluate_model(ckpt_path, rname)
        
    with open(OUT_DIR / "g1_g2_evaluation_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print("\nAll 6 models evaluated successfully!")

