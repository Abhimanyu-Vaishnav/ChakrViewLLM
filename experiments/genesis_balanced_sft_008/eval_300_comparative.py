"""
Mission CV-GENESIS-NR-009: 300-Prompt Comparative Evaluation Suite.
Standardized blind evaluation comparing:
1. NR-005 SFT Best (genesis_sft_best.pt)
2. NR-005 SFT Final (genesis_sft_final.pt)
3. NR-008 Balanced SFT Best (balanced_sft_best.pt)
4. NR-008 Balanced SFT Final (balanced_sft_final.pt)

Uses canonical NR-006 blind benchmark prompts (300 prompts across English, Hindi, Hinglish, Code, QA, Reasoning).
Standardized decoding: eos_id=1, temperature=0.7, top_k=20, max_new_tokens=40.
"""

import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[EVAL] Running on device: {device}")

# 1. Load the 300 blind prompts from NR-006 manifest
manifest_path = Path("reports/NR006_BLIND_DATASET_MANIFEST.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)
eval_prompts = manifest_data.get("prompts_inventory", [])
print(f"[EVAL] Loaded {len(eval_prompts)} standardized blind evaluation prompts.")

# 2. Checkpoint registry
checkpoints = {
    "nr005_sft_best": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
    "nr005_sft_final": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt",
    "nr008_sft_best": "experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_best.pt",
    "nr008_sft_final": "experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_final.pt"
}

def generate_text(model, prompt: str, eos_id: int = 1, max_new_tokens: int = 40) -> str:
    p_ids = tok.encode(prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.7
            v, _ = torch.topk(logits, min(20, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tok_val = idx_next.item()
            if tok_val == eos_id:
                break
            generated.append(tok_val)
            idx = torch.cat((idx, idx_next), dim=1)
    return tok.decode(generated, errors="replace").strip()

def score_output(prompt_meta: dict, response: str) -> str:
    resp = response.strip()
    if not resp:
        return "Incomplete"
    # Check repetition
    words = resp.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) > len(set(trigrams)) * 2:
            return "Repetitive"
    
    # Check expected answers if available
    exp = prompt_meta.get("expected") or prompt_meta.get("expected_answers") or []
    if exp:
        if any(e.lower() in resp.lower() for e in exp):
            return "Correct and relevant"
    
    # Surface coherence heuristics
    cat = prompt_meta.get("category", "")
    lang = prompt_meta.get("language", "en")
    
    if cat == "coding" and ("def " in resp or "return" in resp or "class " in resp or "import" in resp):
        return "Partially correct"
        
    if len(resp) >= 8 and any(c.isalpha() for c in resp):
        return "Partially correct"
        
    return "Incorrect"

all_results = {}

for model_name, ckpt_path in checkpoints.items():
    print(f"\n[BENCHMARK] Evaluating {model_name} ({ckpt_path})...")
    cfg = GenesisConfig()
    model = ChakrGenesis(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    
    # In NR-008 we know eos was supervised as 2, in NR-005 as 1
    eos_val = 2 if "008" in model_name else 1
    
    counts = {
        "Correct and relevant": 0,
        "Partially correct": 0,
        "Incorrect": 0,
        "Incomplete": 0,
        "Repetitive": 0
    }
    
    eval_records = []
    
    for item in eval_prompts:
        p = item["prompt"]
        resp = generate_text(model, p, eos_id=eos_val)
        verdict = score_output(item, resp)
        counts[verdict] = counts.get(verdict, 0) + 1
        eval_records.append({
            "id": item.get("id"),
            "category": item.get("category"),
            "language": item.get("language"),
            "prompt": p,
            "response": resp,
            "verdict": verdict
        })
        
    total = len(eval_prompts)
    percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}
    
    print(f"Results for {model_name}:")
    print("  Counts:     ", counts)
    print("  Percentages:", percentages)
    
    all_results[model_name] = {
        "checkpoint": ckpt_path,
        "total_prompts": total,
        "counts": counts,
        "percentages": percentages,
        "evaluations": eval_records
    }

output_path = Path("reports/NR009_300_BENCHMARK_RESULTS.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n[COMPLETE] 300-Prompt Comparative Evaluation saved to {output_path}")
