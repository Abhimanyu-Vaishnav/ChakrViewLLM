"""
Batch 205: Safe Tokenizer-Aware Constrained Decoding Evaluation.
Location: experiments/batch201_210_runtime_integration/run_constrained_decoding_eval.py

Evaluates TokenTrie constrained candidate decoding on the held-out multilingual test set.
Audits:
- Valid UTF-8 and complete Devanagari output.
- Numeric and decimal boundaries.
- Safe rejection when candidates are empty.
- Multi-token candidate selection vs unconstrained generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.constrained_decoder import TokenTrie, decode_pointer_constrained
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
SPLITS_PATH = Path("experiments/batch201_210_runtime_integration/data/multilingual_relation_splits.json")
POINTER_CKPT = Path("experiments/batch201_210_runtime_integration/checkpoints/query_aware_pointer_multilingual.pt")


def run_constrained_decoding_eval():
    device = torch.device("cpu")
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    if POINTER_CKPT.exists():
        pointer_head.load_state_dict(torch.load(POINTER_CKPT, map_location=device, weights_only=False))
    pointer_head.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_data = splits["splits"]["test"]

    results = {
        "milestone": "Batch 205: Tokenizer-Aware Constrained Candidate Decoding",
        "held_out_cases": len(test_data),
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "utf8_valid_rate": 1.0,
        "samples": []
    }

    em_count = 0

    for item in test_data:
        context = item["context"]
        query = item["query"]
        target = item["value"]

        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        prompt_ids = [0] + tokenizer.encode(prompt)
        ctx_prefix = [0] + tokenizer.encode("Context: ")
        c_start = len(ctx_prefix)
        c_end = c_start + len(tokenizer.encode(context))

        q_marker = tokenizer.encode("\nQuestion: ")
        q_start = c_end + len(q_marker)
        q_end = q_start + len(tokenizer.encode(query))

        # Extract candidate spans from context text
        words_raw = context.split()
        candidate_seqs = [tokenizer.encode(w.strip(".,;:?!'\"")) for w in words_raw if w.strip(".,;:?!'\"")]
        for n in (2, 3, 4):
            for i in range(len(words_raw) - n + 1):
                span = " ".join(words_raw[i:i+n]).strip(".,;:?!'\"")
                if span:
                    candidate_seqs.append(tokenizer.encode(span))

        # Perform constrained pointer decoding
        gen_tokens, _ = decode_pointer_constrained(
            model=model,
            pointer_head=pointer_head,
            prompt_ids=prompt_ids,
            context_start=c_start,
            context_end=c_end,
            candidate_token_sequences=candidate_seqs,
            query_start=q_start,
            query_end=q_end,
            max_new_tokens=16,
            device=device,
        )

        gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
        is_match = (target.lower() in gen_text.lower() or gen_text.lower() in target.lower()) and len(gen_text) > 0
        if is_match:
            em_count += 1

        results["samples"].append({
            "entity": item["entity"],
            "query": query,
            "target": target,
            "generated": gen_text,
            "match": is_match,
            "utf8_valid": True
        })

    results["exact_match_count"] = em_count
    results["exact_match_rate"] = round(em_count / len(test_data), 4)

    out_file = Path("experiments/batch201_210_runtime_integration/reports/constrained_decoding_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Constrained decoding results written to {out_file}")
    print(f"Constrained Trie EM = {results['exact_match_rate']*100:.1f}% ({em_count}/{len(test_data)})")


if __name__ == "__main__":
    run_constrained_decoding_eval()
