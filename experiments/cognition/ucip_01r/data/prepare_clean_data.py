import json
from pathlib import Path

def standardize_and_audit():
    train_in = Path("experiments/cognition/ucip_01/data/ucip01_train.jsonl")
    val_in = Path("experiments/cognition/ucip_01/data/ucip01_validation.jsonl")
    
    out_dir = Path("experiments/cognition/ucip_01r/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_out = out_dir / "ucip01r_train.jsonl"
    val_out = out_dir / "ucip01r_validation.jsonl"
    
    def process_file(in_path, out_path):
        records = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                target = d.get("target") or d.get("response") or ""
                target = target.strip()
                if not target:
                    continue
                # Ensure leading space for BPE boundary
                if not target.startswith(" "):
                    target = " " + target
                prompt = d.get("prompt", "").strip()
                if not prompt.endswith("Assistant:"):
                    prompt = prompt + "\nAssistant:"
                records.append({
                    "prompt": prompt,
                    "target": target,
                    "category": d.get("category", "conversation"),
                    "language": d.get("language", "multilingual")
                })
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(records)
        
    n_train = process_file(train_in, train_out)
    n_val = process_file(val_in, val_out)
    
    print(f"Standardized {n_train} valid train records to {train_out}")
    print(f"Standardized {n_val} valid val records to {val_out}")

if __name__ == "__main__":
    standardize_and_audit()
