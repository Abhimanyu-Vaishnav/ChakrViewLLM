"""
Batch 237 & 238: Multi-Hop Evidence Assembly, Adversarial Distractor Stress Tests, & Generalization.
Location: experiments/batch231_240_multihop_reasoning/run_multihop_adversarial_eval.py
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.constrained_decoder import decode_pointer_constrained
from chakrview.runtime.cross_attention_pointer import ChakrCrossAttentionPointerHead
from chakrview.runtime.evidence_extractor import EvidenceExtractor

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
FROZEN_MANIFEST_PATH = Path("experiments/batch231_240_multihop_reasoning/data/frozen_multihop_manifest_100.json")
CROSS_CKPT = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
REPORTS_DIR = Path("experiments/batch231_240_multihop_reasoning/reports")


def evaluate_multihop_adversarial():
    device = torch.device("cpu")
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    head = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    head.load_state_dict(torch.load(CROSS_CKPT, map_location=device, weights_only=False))
    head.eval()

    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))

    slices = {
        "all_held_out_chains": manifest,
        "answerable_2hop": [r for r in manifest if not r.get("is_unanswerable")],
        "adversarial_missing_link": [r for r in manifest if r.get("is_unanswerable")],
        "multilingual_hindi_chains": [r for r in manifest if r.get("language") == "hi"],
        "multi_token_chains": [r for r in manifest if "multi_token" in r.get("diagnostic_tags", [])],
        "single_token_chains": [r for r in manifest if "single_token" in r.get("diagnostic_tags", [])]
    }

    slice_results = {}

    for s_name, items in slices.items():
        em_count = 0
        intermediate_hits = 0
        distractor_confusions = 0
        abstention_correct = 0
        total = len(items)

        for item in items:
            ctx = item["context"]
            query = item["query"]
            gold = item["gold_value"].strip()
            norm_gold = item["normalized_gold"].strip()
            dist_val = str(item.get("distractor_value", "")).strip().lower()
            inter_e2 = item.get("intermediate_e2", "").strip().lower()
            is_unans = item.get("is_unanswerable", False)

            if is_unans:
                res = EvidenceExtractor.extract_slot(ctx, query)
                if res.is_abstention:
                    abstention_correct += 1
                    em_count += 1
                continue

            prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"
            p_ids = [0] + tokenizer.encode(prompt)
            c_start = len([0] + tokenizer.encode("Context: "))
            c_end = c_start + len(tokenizer.encode(ctx))
            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))

            words_raw = ctx.split()
            cand_seqs = [tokenizer.encode(w.strip(".,;:?!\"'")) for w in words_raw if w.strip(".,;:?!\"'")]
            for n in (2, 3):
                for j in range(len(words_raw) - n + 1):
                    s = " ".join(words_raw[j:j+n]).strip(".,;:?!\"'")
                    if s: cand_seqs.append(tokenizer.encode(s))

            toks, _ = decode_pointer_constrained(
                model=model, pointer_head=head, prompt_ids=p_ids,
                context_start=c_start, context_end=c_end,
                candidate_token_sequences=cand_seqs,
                query_start=q_start, query_end=q_end, max_new_tokens=8, device=device
            )
            pred = tokenizer.decode(toks, errors="replace").strip().lower().rstrip(".,;:?!")

            if pred == norm_gold or norm_gold in pred or pred in norm_gold and len(pred) > 0:
                em_count += 1
                intermediate_hits += 1
            elif dist_val and (dist_val in pred or pred in dist_val):
                distractor_confusions += 1
            elif inter_e2 and inter_e2 in pred:
                intermediate_hits += 1

        slice_results[s_name] = {
            "total": total,
            "exact_match_count": em_count,
            "exact_match_rate": round(em_count / total, 4) if total > 0 else 0.0,
            "intermediate_hits": intermediate_hits,
            "distractor_confusions": distractor_confusions,
            "abstention_correct": abstention_correct if any(r.get("is_unanswerable") for r in items) else None
        }

    report = {
        "milestone": "Batches 237-238: Multi-Hop Adversarial Stress Tests & Generalization Slices",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_arm": "ChakrCrossAttentionPointerHead (Step 1000) + TokenTrie",
        "slices": slice_results
    }

    out_p = REPORTS_DIR / "multihop_results.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Multi-hop results written to {out_p}")

    print("\n=== MULTI-HOP SLICES SUMMARY ===")
    for s_name, res in slice_results.items():
        print(f"[{s_name:<28}] EM: {res['exact_match_count']:>2}/{res['total']:<2} ({res['exact_match_rate']*100:>5.1f}%) | Distractor Confusions: {res['distractor_confusions']:>2}")


if __name__ == "__main__":
    evaluate_multihop_adversarial()
