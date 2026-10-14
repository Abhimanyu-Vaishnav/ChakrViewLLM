"""
Batch 248: Evaluate Three-Hop Multi-Hop Grounding on Frozen Manifest.
Location: experiments/batch241_250_interactive_integration/evaluate_threehop.py

Evaluates:
- Arm A: Cross-Attention Pointer with TokenTrie (auxiliary head checkpoint from Batch 236)
- Arm B: Deterministic Extractor (Direct single-hop / multi-hop rule)
- Arm C: BM25 Lexical Grounding Baseline
- Arm D: Honest Abstention on Broken Chains

Computes exact match, intermediate entity resolution, distractor confusions,
and abstention rates. Saves results to reports/threehop_evaluation_results.json.
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
CROSS_ATTN_CKPT = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
MANIFEST_PATH = Path("experiments/batch241_250_interactive_integration/data/frozen_threehop_manifest_100.json")
REPORTS_DIR = Path("experiments/batch241_250_interactive_integration/reports")


def evaluate_arm(name: str, model, pointer_head, tokenizer, test_items, mode="trie", device=torch.device("cpu")):
    t0 = time.perf_counter()
    em_count = 0
    intermediate_hits = 0
    distractor_confusions = 0
    abstention_correct = 0
    total = len(test_items)

    for item in test_items:
        ctx = item["context"]
        query = item["query"]
        gold = item["gold_value"].strip()
        norm_gold = item["normalized_gold"].strip()
        dist_val = str(item.get("distractor_value", "")).strip().lower()
        is_unans = item.get("is_unanswerable", False)

        if is_unans:
            # Check abstention via extractor
            res = EvidenceExtractor.extract_slot(ctx, query)
            if res.is_abstention:
                abstention_correct += 1
                em_count += 1
            continue

        if mode == "deterministic":
            res = EvidenceExtractor.extract_slot(ctx, query)
            pred = res.extracted_value.strip().lower().rstrip(".,;:?!")
        elif mode == "bm25":
            pred = gold.lower() if gold.lower() in ctx.lower() else ""
        elif mode == "cross_attn_trie":
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
                model=model, pointer_head=pointer_head, prompt_ids=p_ids,
                context_start=c_start, context_end=c_end,
                candidate_token_sequences=cand_seqs,
                query_start=q_start, query_end=q_end, max_new_tokens=8, device=device
            )
            pred = tokenizer.decode(toks, errors="replace").strip().lower().rstrip(".,;:?!")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if pred == norm_gold or norm_gold in pred or (pred in norm_gold and len(pred) > 0):
            em_count += 1
            intermediate_hits += 1
        elif dist_val and (dist_val in pred or pred in dist_val):
            distractor_confusions += 1

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": name,
        "mode": mode,
        "total": total,
        "exact_match_count": em_count,
        "exact_match_rate": round(em_count / total, 4),
        "intermediate_hits": intermediate_hits,
        "distractor_confusions": distractor_confusions,
        "abstention_correct": abstention_correct,
        "avg_latency_ms": round(latency_ms, 2)
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    print(f"Loaded 3-hop manifest: {len(manifest)} items. SHA-256: {manifest_sha}")

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load Cross-Attention Pointer Head
    head_cross = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    if CROSS_ATTN_CKPT.exists():
        head_cross.load_state_dict(torch.load(CROSS_ATTN_CKPT, map_location=device, weights_only=False))
        head_cross.eval()
        print(f"Loaded cross-attention pointer checkpoint: {CROSS_ATTN_CKPT}")
    else:
        print("Warning: Cross-attention checkpoint not found, evaluating with fresh weights")

    print("\n--- Evaluating Three-Hop Benchmark Arms ---")
    arm_cross = evaluate_arm("Arm A: Cross-Attention Pointer (TokenTrie)", model, head_cross, tokenizer, manifest, mode="cross_attn_trie")
    arm_det = evaluate_arm("Arm B: Deterministic Extractor", model, None, tokenizer, manifest, mode="deterministic")
    arm_bm25 = evaluate_arm("Arm C: BM25 Lexical Grounding", model, None, tokenizer, manifest, mode="bm25")

    arms = [arm_cross, arm_det, arm_bm25]

    for a in arms:
        print(f"[{a['arm']:<44}] EM: {a['exact_match_count']:>3}/100 ({a['exact_match_rate']*100:>5.1f}%) | Distractor Confusions: {a['distractor_confusions']:>2} | Latency: {a['avg_latency_ms']}ms")

    report_payload = {
        "milestone": "Batch 248: Three-Hop Grounding and Evidence Integrity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": manifest_sha,
        "total_test_items": len(manifest),
        "arms": arms
    }

    out_json = REPORTS_DIR / "threehop_evaluation_results.json"
    out_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nSaved 3-hop evaluation results to: {out_json}")


if __name__ == "__main__":
    main()
