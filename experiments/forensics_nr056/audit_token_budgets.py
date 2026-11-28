import json
import glob
import os

token_audit = {
    "datasets": {},
    "training_runs": {}
}

# 1. Inspect Corpus V3 / NR-051 Pretraining Data
pretrain_051_train = "data/pretraining_nr051/pretrain_train.jsonl"
pretrain_051_val = "data/pretraining_nr051/pretrain_val.jsonl"
sft_051_train = "data/pretraining_nr051/sft_train.jsonl"
sft_051_val = "data/pretraining_nr051/sft_val.jsonl"

# 2. Inspect Corpus V4 / NR-052 Data
pretrain_v4_train = "data/corpus_v4/pretrain_train.jsonl"
pretrain_v4_val = "data/corpus_v4/pretrain_val.jsonl"
sft_v4_train = "data/corpus_v4/sft_train.jsonl"
sft_v4_val = "data/corpus_v4/sft_val.jsonl"

paths = {
    "nr051_pretrain_train": pretrain_051_train,
    "nr051_pretrain_val": pretrain_051_val,
    "nr051_sft_train": sft_051_train,
    "nr051_sft_val": sft_051_val,
    "v4_pretrain_train": pretrain_v4_train,
    "v4_pretrain_val": pretrain_v4_val,
    "v4_sft_train": sft_v4_train,
    "v4_sft_val": sft_v4_val,
}

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

for label, p in paths.items():
    if not os.path.exists(p):
        token_audit["datasets"][label] = {"exists": False}
        continue
    
    line_count = 0
    total_tokens = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            line_count += 1
            obj = json.loads(line)
            txt = obj.get("text", "")
            if not txt and "prompt" in obj and "response" in obj:
                txt = f"<user>\n{obj['prompt']}\n<assistant>\n{obj['response']}"
            tokens = tokenizer.encode(txt)
            total_tokens += len(tokens)
            
    token_audit["datasets"][label] = {
        "exists": True,
        "path": p,
        "sample_count": line_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": round(total_tokens / line_count, 1) if line_count else 0
    }
    print(f"[{label}] samples={line_count:,} tokens={total_tokens:,}")

with open("experiments/forensics_nr056/token_budget_audit_raw.json", "w") as f:
    json.dump(token_audit, f, indent=2)
