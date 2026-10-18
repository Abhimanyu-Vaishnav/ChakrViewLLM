import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts
)

DATA_DIR = Path("experiments/batch107_110_operand_copy_benchmark/data")

def load_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def parse_copy_output(text: str) -> Dict:
    import re
    # Expect A={d} and B={d}
    res = {"a": None, "b": None}
    ma = re.search(r"A\s*=\s*(\d+)", text)
    if ma:
        res["a"] = int(ma.group(1))
    mb = re.search(r"B\s*=\s*(\d+)", text)
    if mb:
        res["b"] = int(mb.group(1))
    return res

def evaluate_copy_model(model: ChakrMicro, tokenizer, records: List[Dict]) -> Dict:
    model.eval()
    total = len(records)
    a_correct = 0
    b_correct = 0
    pair_correct = 0
    generated_list = []
    
    a_probs, b_probs = [], []
    a_ranks, b_ranks = [], []
    
    for r in records:
        a_val, b_val = r["operands"]
        prompt = r["prompt"]
        prompt_ids = [0] + tokenizer.encode(prompt)
        
        # 1. Greedy generation
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(25):
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
        raw_gen = tokenizer.decode(gen_tokens[len(prompt_ids):])
        clean_gen = raw_gen.split("<|im_end|>")[0].strip()
        generated_list.append(clean_gen)
        
        parsed = parse_copy_output(clean_gen)
        is_a = (parsed["a"] == a_val)
        is_b = (parsed["b"] == b_val)
        if is_a:
            a_correct += 1
        if is_b:
            b_correct += 1
        if is_a and is_b:
            pair_correct += 1
            
        # 2. Teacher-forced probability & rank at exact slot positions
        # Prefix for A: prompt + "A="
        prefix_a = prompt + "A="
        toks_a = [0] + tokenizer.encode(prefix_a)
        target_a_tok = 51 + a_val
        with torch.no_grad():
            logits_a = model(torch.tensor([toks_a]))[0, -1]
            probs_a = torch.softmax(logits_a, dim=-1)
            rank_a = (logits_a > logits_a[target_a_tok]).sum().item() + 1
            a_probs.append(probs_a[target_a_tok].item())
            a_ranks.append(rank_a)
            
        # Prefix for B: prompt + f"A={a_val}\nB="
        prefix_b = prompt + f"A={a_val}\nB="
        toks_b = [0] + tokenizer.encode(prefix_b)
        target_b_tok = 51 + b_val
        with torch.no_grad():
            logits_b = model(torch.tensor([toks_b]))[0, -1]
            probs_b = torch.softmax(logits_b, dim=-1)
            rank_b = (logits_b > logits_b[target_b_tok]).sum().item() + 1
            b_probs.append(probs_b[target_b_tok].item())
            b_ranks.append(rank_b)

    import collections
    counts = collections.Counter(generated_list)
    most_common, max_count = counts.most_common(1)[0] if counts else ("", 0)
    
    return {
        "total": total,
        "operand_a_accuracy": round(a_correct / max(1, total), 4),
        "operand_b_accuracy": round(b_correct / max(1, total), 4),
        "exact_pair_accuracy": round(pair_correct / max(1, total), 4),
        "unique_outputs": len(counts),
        "collapse_rate": round(max_count / max(1, total), 4),
        "most_common_output": most_common,
        "mean_target_prob_a": round(float(np.mean(a_probs)), 4),
        "mean_rank_a": round(float(np.mean(a_ranks)), 2),
        "mean_target_prob_b": round(float(np.mean(b_probs)), 4),
        "mean_rank_b": round(float(np.mean(b_ranks)), 2),
        "samples": generated_list[:3],
    }

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"] if "model_state_dict" in base_state else base_state)
    model.eval()
    
    splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    
    results = {}
    print("=== Evaluating Canonical Step-10,000 Checkpoint on Copy Benchmark ===")
    for s_name, recs in splits.items():
        res = evaluate_copy_model(model, tok, recs)
        results[s_name] = res
        print(f"[{s_name:5s}] A_acc={res['operand_a_accuracy']*100:.1f}%, B_acc={res['operand_b_accuracy']*100:.1f}%, Pair_acc={res['exact_pair_accuracy']*100:.1f}%, Collapse={res['collapse_rate']*100:.1f}%")
        print(f"        Mean P(A)={res['mean_target_prob_a']*100:.2f}% (Rank {res['mean_rank_a']}), Mean P(B)={res['mean_target_prob_b']*100:.2f}% (Rank {res['mean_rank_b']})")
        print(f"        Sample out: {repr(res['samples'][0])}")

    out_p = Path("experiments/batch107_110_operand_copy_benchmark/reports/canonical_baseline_eval.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved canonical baseline results to {out_p}")

if __name__ == "__main__":
    main()
