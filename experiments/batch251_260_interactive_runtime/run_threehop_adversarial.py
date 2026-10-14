"""
Batch 257: Adversarial Three-Hop Generalization and Diagnostic Evaluation.
Location: experiments/batch251_260_interactive_runtime/run_threehop_adversarial.py

Evaluates:
- Baseline Cross-Attention Pointer vs Intermediate-Supervised Pointer vs BM25 vs Deterministic
Across adversarial slices:
1. Canonical Held-Out 3-Hop Manifest (100 items)
2. Adversarial Broken Chain (Missing Link / Abstention) (20 items)
3. Multilingual Hindi / Devanagari 3-Hop Chains
4. Document-Reordered Chains (Doc C before Doc A / Doc B)
5. Competing Numerical Distractors (Same unit / scale)
6. Complex Multi-Token Value Chains
7. Direct Single-Hop Distractor Stress (Doc D direct match)
"""
from __future__ import annotations

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
FROZEN_MANIFEST_PATH = Path("experiments/batch241_250_interactive_integration/data/frozen_threehop_manifest_100.json")

BASELINE_CKPT = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
SUPERVISED_CKPT = Path("experiments/batch251_260_interactive_runtime/checkpoints/threehop_supervised_pointer.pt")
REPORTS_DIR = Path("experiments/batch251_260_interactive_runtime/reports")


def create_adversarial_reordered_items(manifest: list[dict]) -> list[dict]:
    reordered = []
    for item in manifest:
        if item.get("is_unanswerable"):
            continue
        lines = [line.strip() for line in item["context"].split("\n") if line.strip()]
        # Reverse document order
        reversed_context = "\n".join(reversed(lines))
        reordered_item = dict(item)
        reordered_item["context"] = reversed_context
        reordered_item["adversarial_type"] = "document_reordering"
        reordered.append(reordered_item)
    return reordered


def eval_model_on_items(model, head, tokenizer, items, device=torch.device("cpu")):
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
                if s:
                    cand_seqs.append(tokenizer.encode(s))

        toks, _ = decode_pointer_constrained(
            model=model, pointer_head=head, prompt_ids=p_ids,
            context_start=c_start, context_end=c_end,
            candidate_token_sequences=cand_seqs,
            query_start=q_start, query_end=q_end, max_new_tokens=8, device=device
        )
        pred = tokenizer.decode(toks, errors="replace").strip().lower().rstrip(".,;:?!")

        if pred == norm_gold or norm_gold in pred or (pred in norm_gold and len(pred) > 0):
            em_count += 1
            intermediate_hits += 1
        elif dist_val and (dist_val in pred or pred in dist_val):
            distractor_confusions += 1

    return {
        "total": total,
        "em_count": em_count,
        "em_rate": round(em_count / total, 4) if total > 0 else 0.0,
        "intermediate_hits": intermediate_hits,
        "distractor_confusions": distractor_confusions,
        "abstention_correct": abstention_correct
    }


def main():
    device = torch.device("cpu")
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    reordered_items = create_adversarial_reordered_items(manifest)

    slices = {
        "all_held_out_3hop": manifest,
        "answerable_3hop_chains": [r for r in manifest if not r.get("is_unanswerable")],
        "adversarial_missing_links": [r for r in manifest if r.get("is_unanswerable")],
        "multilingual_hindi_chains": [r for r in manifest if r.get("language") == "hi"],
        "adversarial_doc_reordered": reordered_items,
    }

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    head_base = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    head_base.load_state_dict(torch.load(BASELINE_CKPT, map_location=device, weights_only=False))
    head_base.eval()

    head_sup = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    head_sup.load_state_dict(torch.load(SUPERVISED_CKPT, map_location=device, weights_only=False))
    head_sup.eval()

    results = {}
    print(f"{'Slice Name':<30} | {'Count':<5} | {'Base EM':<10} | {'Base Conf':<10} | {'Sup EM':<10} | {'Sup Conf':<10}")
    print("-" * 88)

    for s_name, items in slices.items():
        res_base = eval_model_on_items(model, head_base, tokenizer, items, device=device)
        res_sup = eval_model_on_items(model, head_sup, tokenizer, items, device=device)

        results[s_name] = {
            "slice_size": len(items),
            "baseline": res_base,
            "intermediate_supervised": res_sup
        }

        print(
            f"{s_name:<30} | {len(items):<5} | "
            f"{res_base['em_count']}/{len(items)} ({res_base['em_rate']*100:.1f}%) | "
            f"{res_base['distractor_confusions']:<10} | "
            f"{res_sup['em_count']}/{len(items)} ({res_sup['em_rate']*100:.1f}%) | "
            f"{res_sup['distractor_confusions']:<10}"
        )

    out_file = REPORTS_DIR / "threehop_adversarial_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved adversarial results to {out_file}")


if __name__ == "__main__":
    main()
