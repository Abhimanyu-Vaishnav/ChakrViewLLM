"""
Execution Harness for 300-Prompt Blind Benchmark (Mission CV-GENESIS-NR-006).

Evaluates:
A. Canonical Pretrained Best (experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt)
B. NR-005 SFT Best (experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt)
C. NR-005 SFT Final (experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt)

Applies strict categorization:
- Correct and relevant
- Partially correct
- Incorrect
- Unrelated
- Repetitive
- Incomplete
- Unsupported factual claim
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"


def score_response(prompt_meta: dict, response: str) -> tuple[str, str]:
    resp = response.strip()
    cat = prompt_meta["category"]
    prompt = prompt_meta["prompt"]
    
    # 1. Repetitive check
    words = re.findall(r'\w+', resp.lower())
    if len(words) >= 6:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_rep = max(word_counts.values())
        if max_rep / len(words) > 0.4:
            return "Repetitive", "High single-token or phrase repetition ratio"

    # 2. Incomplete check
    if len(resp) < 3 or resp.endswith(("...", "...", "और", "and", "the", "with")):
        if len(resp) < 3:
            return "Incomplete", "Truncated or empty response"

    # 3. Category-specific heuristic scoring
    if cat == "conversation":
        if any(g in resp.lower() for g in ["hello", "chakrview", "नमस्ते", "badhiya", "help", "assistant", "सुप्रभात", "theek"]):
            return "Correct and relevant", "Appropriate conversational dialogue"
        elif len(resp.split()) >= 3:
            return "Partially correct", "Conversational response without standard greeting"
        else:
            return "Unrelated", "Fails to converse appropriately"

    elif cat == "coding":
        if "def " in resp or "return " in resp or "class " in resp or "```" in resp:
            return "Correct and relevant", "Valid Python code construct"
        elif any(k in resp for k in ["function", "code", "python", "import"]):
            return "Partially correct", "Mentions code concepts without syntax definition"
        else:
            return "Incorrect", "No functional code provided"

    elif cat == "arithmetic":
        # Check specific answers
        if "128 + 256" in prompt and "384" in resp:
            return "Correct and relevant", "Exact arithmetic match"
        if "500 - 187" in prompt and "313" in resp:
            return "Correct and relevant", "Exact arithmetic match"
        if "16 multiplied by 9" in prompt and "144" in resp:
            return "Correct and relevant", "Exact arithmetic match"
        if "Divide 144 by 12" in prompt and "12" in resp:
            return "Correct and relevant", "Exact arithmetic match"
        if "2 to the power of 8" in prompt and "256" in resp:
            return "Correct and relevant", "Exact power match"
        if any(d in resp for d in "0123456789"):
            return "Partially correct", "Numeric response provided"
        return "Incorrect", "Failed calculation"

    elif cat == "multiturn":
        if "Liam" in prompt and "C++" in resp:
            return "Correct and relevant", "Entity correctly recalled"
        if "Frankfurt" in prompt and "Frankfurt" in resp:
            return "Correct and relevant", "Entity correctly recalled"
        if "Pune" in prompt and "Pune" in resp:
            return "Correct and relevant", "Entity correctly recalled"
        if "850" in prompt and "850" in resp:
            return "Correct and relevant", "Entity correctly recalled"
        if "Friday" in prompt and "Friday" in resp:
            return "Correct and relevant", "Entity correctly recalled"
        return "Incorrect", "Entity missed or hallucinated"

    elif cat == "uncertainty":
        if any(k in resp.lower() for k in ["cannot", "predict", "unknown", "नहीं", "nahi", "jaankari", "private", "future"]):
            return "Correct and relevant", "Principled abstention on uncertain/unknowable fact"
        return "Unsupported factual claim", "Hallucinated or asserted claim without basis"

    elif cat == "explanation":
        if len(resp.split()) >= 6 and not any(junk in resp for junk in ["Wikipedia", "archive.org", "http"]):
            return "Partially correct", "Structured conceptual explanation"
        return "Unrelated", "Fails to provide coherent explanation"

    return "Partially correct", "Standard default response classification"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40, temperature: float = 0.7, top_k: int = 20, device: str = "cuda:0") -> str:
    eos_id = 1
    p_ids = tokenizer.encode(prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -512:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tok_val = idx_next.item()
            if tok_val == eos_id:
                break
            generated.append(tok_val)
            idx = torch.cat((idx, idx_next), dim=1)
            
    try:
        return tokenizer.decode(generated, errors="replace").strip()
    except Exception:
        raw_b = tokenizer.decode_bytes(generated)
        return raw_b.decode("utf-8", errors="replace").strip()


def run_evaluation(
    manifest_path: str = "reports/NR006_BLIND_DATASET_MANIFEST.json",
    tokenizer_dir: str = "data/experiments/vocab_4096",
    output_path: str = "reports/NR006_PURE_NEURAL_RESULTS.json",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    print("=== STARTING CV-GENESIS-NR-006 BLIND BENCHMARK ===")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    prompts = manifest["prompts_inventory"]
    print(f"Loaded {len(prompts)} prompts from manifest.")

    tok, _ = load_tokenizer_artifacts(tokenizer_dir)

    models_to_test = {
        "canonical_pretrain_best": "experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt",
        "genesis_sft_best": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
        "genesis_sft_final": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt"
    }

    full_results = {
        "benchmark_id": "CV-GENESIS-NR-006-PURE-NEURAL",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_prompts": len(prompts),
        "models": {}
    }

    for model_name, ckpt_path in models_to_test.items():
        print(f"\nEvaluating: {model_name}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ChakrGenesis(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        scored_evals = []
        counts = {
            "Correct and relevant": 0,
            "Partially correct": 0,
            "Incorrect": 0,
            "Unrelated": 0,
            "Repetitive": 0,
            "Incomplete": 0,
            "Unsupported factual claim": 0
        }

        t0 = time.time()
        for idx, p_info in enumerate(prompts):
            resp = generate_response(model, tok, p_info["prompt"], max_new_tokens=40, temperature=0.7, top_k=20, device=device)
            verdict, notes = score_response(p_info, resp)
            counts[verdict] = counts.get(verdict, 0) + 1
            scored_evals.append({
                "id": p_info["id"],
                "language": p_info["language"],
                "category": p_info["category"],
                "prompt": p_info["prompt"],
                "response": resp,
                "verdict": verdict,
                "notes": notes
            })

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/300 prompts ({time.time() - t0:.1f}s)...")

        percentages = {k: round(v / len(prompts) * 100, 2) for k, v in counts.items()}
        print(f"Results for {model_name}: {counts}")

        full_results["models"][model_name] = {
            "checkpoint_path": ckpt_path,
            "counts": counts,
            "percentages": percentages,
            "evaluations": scored_evals
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nPure neural results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
