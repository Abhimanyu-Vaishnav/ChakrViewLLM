import json
import collections

def analyze_jsonl(path):
    total = 0
    categories = collections.Counter()
    prompt_lens = []
    target_lens = []
    unique_prompts = set()
    unique_targets = set()
    target_tokens = collections.Counter()
    keys_seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            keys_seen.update(record.keys())
            cat = record.get("category", record.get("domain", record.get("task", "unknown")))
            categories[cat] += 1
            prompt = record.get("prompt", record.get("input", ""))
            target = record.get("target", record.get("completion", record.get("output", "")))
            
            unique_prompts.add(prompt)
            unique_targets.add(target)
            
            prompt_lens.append(len(prompt.split()))
            target_lens.append(len(target.split()))

            for word in target.lower().split():
                target_tokens[word] += 1

    return {
        "path": path,
        "total": total,
        "keys": list(keys_seen),
        "categories": dict(categories.most_common()),
        "unique_prompts": len(unique_prompts),
        "unique_targets": len(unique_targets),
        "avg_prompt_len": sum(prompt_lens) / max(len(prompt_lens), 1),
        "avg_target_len": sum(target_lens) / max(len(target_lens), 1),
        "top_target_tokens": target_tokens.most_common(30)
    }

def main():
    u09_train = analyze_jsonl("experiments/cognition/ucip_09/data/ucip09_train.jsonl")
    u09_val = analyze_jsonl("experiments/cognition/ucip_09/data/ucip09_validation.jsonl")
    u09r_train = analyze_jsonl("experiments/cognition/ucip_09r/data/ucip09r_train.jsonl")
    u09r_val = analyze_jsonl("experiments/cognition/ucip_09r/data/ucip09r_val.jsonl")

    res = {
        "ucip09_train": u09_train,
        "ucip09_val": u09_val,
        "ucip09r_train": u09r_train,
        "ucip09r_val": u09r_val,
    }

    with open("experiments/ucip09s/evaluations/dataset_forensic_stats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("=== UCIP-09 TRAIN ===")
    print(f"Total: {u09_train['total']}, Unique Prompts: {u09_train['unique_prompts']}, Unique Targets: {u09_train['unique_targets']}")
    print(f"Categories: {u09_train['categories']}")
    print(f"Top target tokens: {u09_train['top_target_tokens'][:15]}")

    print("\n=== UCIP-09R TRAIN ===")
    print(f"Total: {u09r_train['total']}, Unique Prompts: {u09r_train['unique_prompts']}, Unique Targets: {u09r_train['unique_targets']}")
    print(f"Categories: {u09r_train['categories']}")
    print(f"Top target tokens: {u09r_train['top_target_tokens'][:15]}")

if __name__ == "__main__":
    main()
