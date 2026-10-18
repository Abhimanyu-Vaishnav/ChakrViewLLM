import json
import collections
from pathlib import Path
import numpy as np
import torch

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch107_110_operand_copy_benchmark.eval_canonical_baseline import (
    DATA_DIR, load_records, parse_copy_output
)

def evaluate_free_running(model: ChakrMicro, tokenizer, records: list) -> dict:
    model.eval()
    total = len(records)
    a_correct = 0
    b_correct = 0
    pair_correct = 0
    detailed_records = []
    generated_list = []
    
    for r in records:
        a_val, b_val = r["operands"]
        prompt = r["prompt"]
        prompt_ids = [0] + tokenizer.encode(prompt)
        
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
        is_pair = (is_a and is_b)
        
        if is_a:
            a_correct += 1
        if is_b:
            b_correct += 1
        if is_pair:
            pair_correct += 1
            
        detailed_records.append({
            "operands": [a_val, b_val],
            "raw": clean_gen,
            "parsed": parsed,
            "correct": is_pair,
        })
        
    counts = collections.Counter(generated_list)
    most_common, max_count = counts.most_common(1)[0] if counts else ("", 0)
    
    return {
        "total": total,
        "a_acc": round(a_correct / max(1, total), 4),
        "b_acc": round(b_correct / max(1, total), 4),
        "pair_acc": round(pair_correct / max(1, total), 4),
        "unique_outputs": len(counts),
        "collapse_rate": round(max_count / max(1, total), 4),
        "most_common": most_common,
        "detailed": detailed_records,
    }

def evaluate_constant_baseline(records: list, const_pair=(1, 8)) -> dict:
    total = len(records)
    a_correct = 0
    b_correct = 0
    pair_correct = 0
    detailed_records = []
    gen_text = f"A={const_pair[0]}\nB={const_pair[1]}"
    
    for r in records:
        a_val, b_val = r["operands"]
        is_a = (const_pair[0] == a_val)
        is_b = (const_pair[1] == b_val)
        is_pair = (is_a and is_b)
        if is_a: a_correct += 1
        if is_b: b_correct += 1
        if is_pair: pair_correct += 1
        detailed_records.append({
            "operands": [a_val, b_val],
            "raw": gen_text,
            "parsed": {"a": const_pair[0], "b": const_pair[1]},
            "correct": is_pair,
        })
    return {
        "total": total,
        "a_acc": round(a_correct / max(1, total), 4),
        "b_acc": round(b_correct / max(1, total), 4),
        "pair_acc": round(pair_correct / max(1, total), 4),
        "unique_outputs": 1,
        "collapse_rate": 1.0,
        "most_common": gen_text,
        "detailed": detailed_records,
    }

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    
    report = {}
    
    # 1. Constant Baseline (Deliberately outputs A=1, B=8)
    print("\n--- 1. Evaluating Constant Baseline (A=1, B=8) ---")
    const_res = {s: evaluate_constant_baseline(splits[s]) for s in ["train", "val", "test"]}
    for s, v in const_res.items():
        print(f"[{s:5s}] pair_acc={v['pair_acc']*100:.1f}%, collapse={v['collapse_rate']*100:.1f}%, unique={v['unique_outputs']}")
    report["constant_baseline"] = const_res
    
    # 2. Canonical Step-10k Checkpoint
    print("\n--- 2. Evaluating Canonical Step-10k Checkpoint ---")
    cfg = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(cfg)
    st_base = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(st_base["model_state_dict"] if "model_state_dict" in st_base else st_base)
    base_res = {s: evaluate_free_running(base_model, tok, splits[s]) for s in ["train", "val", "test"]}
    for s, v in base_res.items():
        print(f"[{s:5s}] pair_acc={v['pair_acc']*100:.1f}%, collapse={v['collapse_rate']*100:.1f}%, unique={v['unique_outputs']}")
    report["canonical_step10k"] = base_res
    
    # 3. Reload Copy Task Checkpoint
    copy_ckpt_path = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")
    sha_copy = compute_file_sha256(str(copy_ckpt_path))
    print(f"\n--- 3. Evaluating Copy Task Checkpoint (SHA-256: {sha_copy}) ---")
    copy_model = ChakrMicro(cfg)
    st_copy = torch.load(copy_ckpt_path, map_location="cpu", weights_only=False)
    copy_model.load_state_dict(st_copy["model_state_dict"])
    copy_res = {s: evaluate_free_running(copy_model, tok, splits[s]) for s in ["train", "val", "test"]}
    for s, v in copy_res.items():
        print(f"[{s:5s}] pair_acc={v['pair_acc']*100:.1f}%, collapse={v['collapse_rate']*100:.1f}%, unique={v['unique_outputs']}")
    report["copy_task_checkpoint"] = copy_res
    
    out_p = Path("experiments/batch107_110_operand_copy_benchmark/reports/reproduction_audit_report.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved reproduction audit report to {out_p}")

if __name__ == "__main__":
    main()
