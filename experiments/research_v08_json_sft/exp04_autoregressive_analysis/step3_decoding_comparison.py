"""Step 3: Decoding Comparisons on Fixed Diagnostic & Held-Out Test Subsets.

Compares:
1. Greedy decoding (argmax, beam_width=1).
2. Deterministic Beam Search (beam_width=2, beam_width=4).
3. Temperature Sampling with nucleus top-p (T=0.3, top_p=0.9, 3 seeds).

Evaluates on:
- Exploratory diagnostic set (Train, Val, Test items).
- Preserved separate final evaluation test subset (10 separate held-out records).

Measures:
- Exact string match.
- Schema match.
- Output text.
- Stop reason.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp02_string_copy.evaluate_exp02 import (
    generate_greedy,
    evaluate_output_case,
)

OUTPUT_FILE = Path(__file__).resolve().parent / "step3_decoding_comparison.json"


def beam_search_deterministic(
    model: ChakrMicro,
    tokenizer: Any,
    input_tokens: List[int],
    beam_width: int = 4,
    max_new_tokens: int = 32,
    context_limit: int = 256,
) -> Dict[str, Any]:
    # Each beam: (tokens_list, cumulative_log_prob)
    beams = [([], 0.0)]
    prefix = list(input_tokens)

    for step_idx in range(max_new_tokens):
        all_candidates = []
        all_completed = True

        for tokens, score in beams:
            # If beam already ended with EOS (1), preserve it
            if len(tokens) > 0 and tokens[-1] == 1:
                all_candidates.append((tokens, score))
                continue

            all_completed = False
            curr = prefix + tokens
            if len(curr) >= context_limit:
                all_candidates.append((tokens, score))
                continue

            x = torch.tensor([curr], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_inds = torch.topk(log_probs, beam_width)
            for lp, ind in zip(topk_log_probs.tolist(), topk_inds.tolist()):
                all_candidates.append((tokens + [ind], score + lp))

        if all_completed:
            break

        # Sort all candidates by score (length-normalized)
        # score / (len(tokens) ** 0.7)
        def length_penalty(tokens: List[int]) -> float:
            return max(1.0, float(len(tokens))) ** 0.7

        all_candidates.sort(key=lambda b: b[1] / length_penalty(b[0]), reverse=True)
        beams = all_candidates[:beam_width]

    best_tokens, best_score = beams[0]
    # Filter out EOS if present at end for decoding
    dec_tokens = [t for t in best_tokens if t != 1]
    out_text = tokenizer.decode(dec_tokens)

    return {
        "text": out_text,
        "tokens": best_tokens,
        "score": round(best_score, 4),
        "stop_reason": "EOS" if (best_tokens and best_tokens[-1] == 1) else "MAX_TOKENS",
    }


def sample_with_temperature(
    model: ChakrMicro,
    tokenizer: Any,
    input_tokens: List[int],
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    curr = list(input_tokens)
    gen_tokens = []

    for _ in range(max_new_tokens):
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1, :] / max(1e-5, temperature)
        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

        filtered_probs = F.softmax(logits, dim=-1)
        next_tok = int(torch.multinomial(filtered_probs, num_samples=1).item())

        if next_tok == 1:
            break
        gen_tokens.append(next_tok)
        curr.append(next_tok)

    return {
        "text": tokenizer.decode(gen_tokens),
        "tokens": gen_tokens,
        "seed": seed,
    }


def run_decoding_comparisons():
    print("=" * 70)
    print("STEP 3: DECODING COMPARISONS ACROSS SAMPLING & BEAM SEARCH")
    print("=" * 70)

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    exp02_path = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt"
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    st = torch.load(exp02_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    test_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "test.jsonl"
    with open(test_file, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if "single_field" in line]

    # Partition test into exploratory diagnostic subset (first 10) and separate evaluation subset (last 10)
    diagnostic_subset = test_records[:10]
    final_eval_subset = test_records[10:20]

    decoding_configs = [
        {"name": "Greedy (argmax)", "type": "greedy"},
        {"name": "Beam Search (k=2)", "type": "beam", "k": 2},
        {"name": "Beam Search (k=4)", "type": "beam", "k": 4},
        {"name": "Sampling (T=0.3, p=0.9, seed=101)", "type": "sample", "temp": 0.3, "p": 0.9, "seed": 101},
        {"name": "Sampling (T=0.3, p=0.9, seed=202)", "type": "sample", "temp": 0.3, "p": 0.9, "seed": 202},
        {"name": "Sampling (T=0.7, p=0.9, seed=303)", "type": "sample", "temp": 0.7, "p": 0.9, "seed": 303},
    ]

    all_results = {}

    for split_label, dataset in [("Exploratory_Diagnostic", diagnostic_subset), ("Preserved_Evaluation", final_eval_subset)]:
        print(f"\n--- Running on {split_label} ({len(dataset)} items) ---")
        split_summary = {}

        for cfg in decoding_configs:
            cfg_name = cfg["name"]
            cfg_type = cfg["type"]
            exact_matches = 0
            schema_valid = 0
            json_parsed = 0
            examples = []

            for rec in dataset:
                p = rec["prompt"] if rec["prompt"].endswith("\n") else rec["prompt"].strip()
                expected_status = rec["expected_status"]
                p_toks = tok.encode(p)
                inp = [0] + p_toks

                if cfg_type == "greedy":
                    out = generate_greedy(model, tok, inp, max_new_tokens=32, context_limit=256)
                    txt = out["text"]
                elif cfg_type == "beam":
                    out = beam_search_deterministic(model, tok, inp, beam_width=cfg["k"], max_new_tokens=32, context_limit=256)
                    txt = out["text"]
                elif cfg_type == "sample":
                    out = sample_with_temperature(model, tok, inp, temperature=cfg["temp"], top_p=cfg["p"], max_new_tokens=32, seed=cfg["seed"])
                    txt = out["text"]

                ev = evaluate_output_case("single_field_string_extraction", txt, expected_status=expected_status)
                if ev["json_parsed"]:
                    json_parsed += 1
                if ev["schema_valid"]:
                    schema_valid += 1
                if ev["status_match"]:
                    exact_matches += 1

                examples.append({
                    "record_id": rec["id"],
                    "status_value": expected_status,
                    "output_text": txt,
                    "status_match": ev["status_match"],
                })

            stat_res = {
                "json_parsed": json_parsed,
                "schema_valid": schema_valid,
                "exact_matches": exact_matches,
                "total": len(dataset),
                "examples": examples,
            }
            split_summary[cfg_name] = stat_res
            print(f"  {cfg_name:35s} | Parsed: {json_parsed:2d}/{len(dataset)} | Schema: {schema_valid:2d}/{len(dataset)} | Match: {exact_matches:2d}/{len(dataset)}")

        all_results[split_label] = split_summary

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDecoding comparison results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_decoding_comparisons()
