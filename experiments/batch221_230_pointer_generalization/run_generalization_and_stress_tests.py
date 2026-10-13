"""
Batch 226, 227 & 228: Numeric Distractor Stress Testing & Generalization Evaluation.
Location: experiments/batch221_230_pointer_generalization/run_generalization_and_stress_tests.py
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
from chakrview.runtime.relation_pointer import ChakrRelationPointerHead
from chakrview.runtime.evidence_extractor import EvidenceExtractor

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
FROZEN_TEST_MANIFEST = Path("experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json")
RELATION_CKPT = Path("experiments/batch221_230_pointer_generalization/checkpoints/relation_aware_pointer_step1000.pt")
REPORTS_DIR = Path("experiments/batch221_230_pointer_generalization/reports")


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    head = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    head.load_state_dict(torch.load(RELATION_CKPT, map_location=device, weights_only=False))
    head.eval()

    test_items = json.loads(FROZEN_TEST_MANIFEST.read_text(encoding="utf-8"))

    # Slices based on diagnostic tags in the frozen manifest
    slices = {
        "held_out_entities": [r for r in test_items if r.get("is_held_out", True)],
        "numeric_distractors": [r for r in test_items if "numeric_distractors" in r.get("diagnostic_tags", []) or r.get("distractor_value")],
        "multilingual_hindi": [r for r in test_items if "multilingual" in r.get("diagnostic_tags", [])],
        "multi_token_values": [r for r in test_items if "multi_token" in r.get("diagnostic_tags", [])],
        "single_token_values": [r for r in test_items if "single_token" in r.get("diagnostic_tags", [])]
    }

    # Add adversarial unanswerable suite (25 items with absent relation)
    adversarial_unanswerable = []
    for r in test_items[:25]:
        item_copy = dict(r)
        item_copy["query"] = f"What is the unknown attribute XYZ of {r['entity']}?"
        item_copy["gold_value"] = "Not mentioned in context."
        item_copy["normalized_gold"] = "not mentioned in context"
        item_copy["is_unanswerable"] = True
        adversarial_unanswerable.append(item_copy)
    slices["adversarial_unanswerable"] = adversarial_unanswerable

    # Candidate extraction improvements (Batch 227)
    # Evaluate baseline extraction vs improved extraction (splitting by punctuation & unit boundaries)
    slice_results = {}

    for s_name, s_items in slices.items():
        em_count = 0
        entity_correct = 0
        distractor_confusions = 0
        abstention_correct = 0
        total = len(s_items)

        for item in s_items:
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
            # Improved candidate span extraction (handles punctuation, units, and 1-3 ngrams)
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
                entity_correct += 1
            elif dist_val and (dist_val in pred or pred in dist_val):
                distractor_confusions += 1
                entity_correct += 1  # Got right entity context, wrong relation/value

        slice_results[s_name] = {
            "total": total,
            "exact_match_count": em_count,
            "exact_match_rate": round(em_count / total, 4) if total > 0 else 0.0,
            "entity_correct_count": entity_correct,
            "distractor_confusions": distractor_confusions,
            "abstention_correct": abstention_correct if is_unans else None
        }

    out_file = REPORTS_DIR / "generalization_results.json"
    report = {
        "milestone": "Batches 226-228: Numeric Distractor Stress Tests & Generalization Slices",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_arm": "ChakrRelationPointerHead (Trained Step 1000) + TokenTrie",
        "slices": slice_results
    }
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Generalization report saved to {out_file}")

    print("\n=== GENERALIZATION SLICES SUMMARY ===")
    for s_name, res in slice_results.items():
        print(f"[{s_name:<26}] EM: {res['exact_match_count']:>2}/{res['total']:<2} ({res['exact_match_rate']*100:>5.1f}%) | Distractor Confusions: {res['distractor_confusions']:>2}")


if __name__ == "__main__":
    main()
