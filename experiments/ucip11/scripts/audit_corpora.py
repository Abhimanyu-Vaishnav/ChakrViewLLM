"""
Fast Corpus Auditor for Phase B.
Samples lines and computes exact statistics without quadratic text concatenation.
"""

import sys
import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

corpus_groups = {
    "stage_b_train": REPO_ROOT / "data" / "processed" / "stage_b" / "train",
    "stage_b_val": REPO_ROOT / "data" / "processed" / "stage_b" / "validation",
    "stage_b_test": REPO_ROOT / "data" / "processed" / "stage_b" / "test",
    "sft_exp34_v3_train": REPO_ROOT / "data" / "sft" / "exp34_candidate_v3" / "splits" / "exp34_train_split_900.jsonl",
    "sft_exp34_v3_val": REPO_ROOT / "data" / "sft" / "exp34_candidate_v3" / "splits" / "exp34_val_split_100.jsonl",
    "sft_m3_train": REPO_ROOT / "data" / "sft" / "m3_conversational" / "splits" / "m3_train_split_540.jsonl",
    "sft_m3_val": REPO_ROOT / "data" / "sft" / "m3_conversational" / "splits" / "m3_val_split_60.jsonl",
    "ucip09_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl",
    "ucip09_val": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_validation.jsonl"
}

summary_stats = {}

for name, path in corpus_groups.items():
    if not path.exists():
        continue
    
    entries = []
    total_bytes = 0
    
    if path.is_file():
        total_bytes = path.stat().st_size
        content = path.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                p = data.get("prompt", "")
                r = data.get("target", data.get("response", ""))
                entries.append((p, r))
            except Exception:
                entries.append((line, ""))
    elif path.is_dir():
        for f in path.glob("*.txt"):
            total_bytes += f.stat().st_size
            lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            for l in lines:
                entries.append((l, ""))
                
    total_entries = len(entries)
    unique_prompts = set(e[0] for e in entries)
    unique_pairs = set(entries)
    
    # Sample tokenization for fertility and token count
    # Tokenize up to 200 items to estimate average tokens per entry
    sample_size = min(200, total_entries)
    sampled = entries[:sample_size]
    sample_tokens = 0
    sample_chars = 0
    for p, r in sampled:
        txt = f"{p} {r}".strip()
        sample_chars += len(txt)
        toks = tokenizer.encode(txt, add_bos=False, add_eos=False)
        sample_tokens += len(toks)
        
    avg_tokens = sample_tokens / sample_size if sample_size > 0 else 0
    avg_chars = sample_chars / sample_size if sample_size > 0 else 0
    est_total_tokens = int(avg_tokens * total_entries)
    
    summary_stats[name] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "total_bytes": total_bytes,
        "total_entries": total_entries,
        "unique_entries": len(unique_pairs),
        "duplicate_rate_pct": ((total_entries - len(unique_pairs)) / total_entries * 100.0) if total_entries > 0 else 0,
        "avg_tokens_per_entry": round(avg_tokens, 1),
        "avg_chars_per_entry": round(avg_chars, 1),
        "estimated_total_tokens": est_total_tokens
    }

out_p = REPO_ROOT / "experiments" / "ucip11" / "data" / "manifests" / "corpus_audit_summary.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2)

print("Corpus audit summary saved.")
for k, v in summary_stats.items():
    print(f"[{k}] {v['total_entries']} entries | {v['unique_entries']} unique (Dup: {v['duplicate_rate_pct']:.1f}%) | Est Tokens: {v['estimated_total_tokens']:,}")
