import json
import glob
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

VOCAB_DIR = "data/experiments/vocab_4096"
DATA_DIR = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/data"

def run_audit():
    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)
    splits = ["train", "val", "test_1", "test_2a", "test_2b", "test_2c"]
    data = {}
    for s in splits:
        with open(f"{DATA_DIR}/{s}.jsonl", encoding="utf-8") as f:
            data[s] = [json.loads(line) for line in f]

    # 1. Exact prompt duplicates across splits
    train_prompts = set(r["prompt"] for r in data["train"])
    val_prompts = set(r["prompt"] for r in data["val"])

    overlaps = {}
    for s in ["val", "test_1", "test_2a", "test_2b", "test_2c"]:
        s_prompts = set(r["prompt"] for r in data[s])
        overlaps[f"train_vs_{s}_exact_prompt_overlap"] = len(train_prompts.intersection(s_prompts))
        overlaps[f"val_vs_{s}_exact_prompt_overlap"] = len(val_prompts.intersection(s_prompts))

    # 2. Normalized prompt duplicates (masking target token)
    def normalize_prompt(p, target):
        return p.replace(f"[{target}]", "[<MASK>]")

    train_norm = set(normalize_prompt(r["prompt"], r["target"]) for r in data["train"])
    norm_overlaps = {}
    for s in ["val", "test_1", "test_2a", "test_2b", "test_2c"]:
        s_norm = set(normalize_prompt(r["prompt"], r["target"]) for r in data[s])
        norm_overlaps[f"train_vs_{s}_template_overlap"] = len(train_norm.intersection(s_norm))
        norm_overlaps[f"{s}_unique_templates"] = len(s_norm)
    norm_overlaps["train_unique_templates"] = len(train_norm)

    # 3. Target token distribution & frequency
    target_distributions = {}
    all_targets = set()
    for s in splits:
        targets = [r["target"] for r in data[s]]
        all_targets.update(targets)
        target_distributions[s] = {
            "unique_targets": len(set(targets)),
            "targets": sorted(list(set(targets))),
            "counts": {t: targets.count(t) for t in sorted(list(set(targets)))}
        }

    # 4. Tokenizer atomicity of every target
    atomicity = {}
    non_atomic = {}
    for t in sorted(list(all_targets)):
        encoded = tok.encode(t)
        atomicity[t] = {
            "length": len(encoded),
            "token_ids": encoded,
            "is_atomic": len(encoded) == 1
        }
        if len(encoded) != 1:
            non_atomic[t] = encoded

    # 5. Held-out categorization
    # Distinguish: Held-out records, Held-out templates, Held-out token identities, Held-out token classes
    train_tokens = set(target_distributions["train"]["targets"])
    test_categories = {}
    for s in ["test_1", "test_2a", "test_2b", "test_2c"]:
        s_tokens = set(target_distributions[s]["targets"])
        unseen_tokens = s_tokens - train_tokens
        seen_tokens = s_tokens.intersection(train_tokens)
        test_categories[s] = {
            "unseen_token_identities": sorted(list(unseen_tokens)),
            "seen_token_identities": sorted(list(seen_tokens)),
            "is_token_identity_held_out": len(seen_tokens) == 0,
            "novelty_criterion": (
                "Held-out template phrasing and stream names; seen token identities"
                if s == "test_1" else
                "Held-out uppercase/lowercase letters (M-P, m-p) not seen in training target"
                if s == "test_2a" else
                "Held-out symbols and punctuation (!, 8, 9, <, >, ?, ^, ~) not seen in training target"
                if s == "test_2b" else
                "Held-out uppercase letters (Q-X) not seen in training target"
            )
        }

    audit_results = {
        "record_counts": {s: len(data[s]) for s in splits},
        "exact_prompt_leakage": overlaps,
        "template_overlap": norm_overlaps,
        "target_distributions": target_distributions,
        "non_atomic_targets": non_atomic,
        "token_atomicity_summary": {
            "total_evaluated_targets": len(all_targets),
            "all_atomic": len(non_atomic) == 0
        },
        "split_integrity_classification": test_categories
    }

    out_path = "experiments/overnight_runs/exp27_forensic_pointer_research_20261025_203725/LEAKAGE_AND_SPLIT_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print("Saved split audit to", out_path)

if __name__ == "__main__":
    run_audit()
