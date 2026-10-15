import json
from pathlib import Path

def update_split(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    updated = []
    for item in lines:
        p = item["prompt"].strip()
        # Ensure standard User: ... \nIntent: prompt framing
        new_prompt = f"User: {p}\nIntent:"
        item["prompt"] = new_prompt
        # Ensure target starts with space for clean BPE continuation
        if not item["target"].startswith(" "):
            item["target"] = " " + item["target"]
        updated.append(item)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in updated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Updated {len(updated)} items in {file_path}")

data_dir = Path("experiments/batch541_580/data")
update_split(data_dir / "train_intent_sft.jsonl")
update_split(data_dir / "val_intent_sft.jsonl")
update_split(data_dir / "test_intent_heldout.jsonl")
