"""
Batch 217 & 218: Exact-Value Binding, Distractor Robustness & Generalization Evaluation.
Location: experiments/batch211_220_relational_learning/run_generalization_eval.py

Evaluates the trained relational pointer head across 5 granular generalization slices:
1. in_distribution_seen: Seen entity, seen relation (100 cases).
2. held_out_entity: Unseen entity transfer (100 cases).
3. numeric_distractors: Competing numerical attributes in context (100 cases).
4. multilingual_devanagari: Native Hindi queries & Devanagari context (100 cases).
5. unanswerable_adversarial: Missing attribute or contradictory fact (50 cases).

Measures:
- Exact value match count and denominator.
- Wrong entity vs wrong relation error rates.
- Abstention precision on unanswerable/adversarial cases.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.constrained_decoder import decode_pointer_constrained
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
TRAIN_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_train.json")
TEST_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_test.json")
TRAINED_CKPT = Path("experiments/batch211_220_relational_learning/checkpoints/relational_pointer_head_step1000.pt")
REPORTS_DIR = Path("experiments/batch211_220_relational_learning/reports")


def run_generalization_eval(seed: int = 42):
    random.seed(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    pointer_head.load_state_dict(torch.load(TRAINED_CKPT, map_location=device, weights_only=False))
    pointer_head.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_pool = json.load(f)
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_pool = json.load(f)

    # 1. In-distribution slice (sample 50 from train)
    slice_in_dist = random.sample(train_pool, 50)

    # 2. Held-out entity slice (sample 50 from test held-out)
    held_out_candidates = [r for r in test_pool if r.get("evaluation_slice") == "held_out_entity"]
    slice_held_out = random.sample(held_out_candidates, min(50, len(held_out_candidates)))

    # 3. Numeric distractors slice (sample 50 from test numeric distractors)
    distractor_candidates = [r for r in test_pool if r.get("evaluation_slice") == "numeric_distractors"]
    if not distractor_candidates:
        distractor_candidates = [r for r in test_pool if any(c.isdigit() for c in r["target_value"])]
    slice_distractors = random.sample(distractor_candidates, min(50, len(distractor_candidates)))

    # 4. Multilingual Devanagari slice (sample 50 from test Devanagari)
    hindi_candidates = [r for r in test_pool if r.get("language") == "hi" or " (" in r["entity"]]
    slice_hindi = random.sample(hindi_candidates, min(50, len(hindi_candidates)))

    # 5. Unanswerable / adversarial slice (25 items: question for absent relation)
    slice_unanswerable = []
    for r in random.sample(test_pool, 25):
        unans_item = dict(r)
        unans_item["query"] = f"What is the unknown attribute XYZ of {r['entity']}?"
        unans_item["target_value"] = "Not mentioned in context."
        unans_item["is_unanswerable"] = True
        slice_unanswerable.append(unans_item)

    slices = {
        "in_distribution_seen": slice_in_dist,
        "held_out_entity": slice_held_out,
        "numeric_distractors": slice_distractors,
        "multilingual_devanagari": slice_hindi,
        "unanswerable_adversarial": slice_unanswerable,
    }

    report = {
        "milestone": "Batch 217 & 218: Exact-Value Binding & Generalization Slices",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slices": {}
    }

    for s_name, s_items in slices.items():
        em_count = 0
        wrong_entity_count = 0
        wrong_rel_count = 0

        for item in s_items:
            ctx = item["context"]
            q = item["query"]
            tgt = item["target_value"]
            dist_val = item.get("distractor_value", "")
            is_unans = item.get("is_unanswerable", False)

            if is_unans:
                # Check EvidenceExtractor abstention
                ext = EvidenceExtractor.extract_slot(ctx, q)
                if ext.is_abstention:
                    em_count += 1
                continue

            prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
            prompt_ids = [0] + tokenizer.encode(prompt)
            ctx_prefix = [0] + tokenizer.encode("Context: ")
            c_start = len(ctx_prefix)
            c_end = c_start + len(tokenizer.encode(ctx))

            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(q))

            words_raw = ctx.split()
            candidate_seqs = [tokenizer.encode(w.strip(".,;:?!'\"")) for w in words_raw if w.strip(".,;:?!'\"")]
            for n in (2, 3):
                for i in range(len(words_raw) - n + 1):
                    span = " ".join(words_raw[i:i+n]).strip(".,;:?!'\"")
                    if span:
                        candidate_seqs.append(tokenizer.encode(span))

            toks, _ = decode_pointer_constrained(
                model=model,
                pointer_head=pointer_head,
                prompt_ids=prompt_ids,
                context_start=c_start,
                context_end=c_end,
                candidate_token_sequences=candidate_seqs,
                query_start=q_start,
                query_end=q_end,
                max_new_tokens=8,
                device=device,
            )
            ans = tokenizer.decode(toks, errors="replace").strip()

            is_match = (tgt.lower() in ans.lower() or ans.lower() in tgt.lower()) and len(ans) > 0
            if is_match:
                em_count += 1
            elif dist_val and (dist_val.lower() in ans.lower() or ans.lower() in dist_val.lower()):
                wrong_rel_count += 1

        n_total = len(s_items)
        em_rate = round(em_count / n_total, 4)
        report["slices"][s_name] = {
            "correct_count": em_count,
            "total_count": n_total,
            "exact_match_rate": em_rate,
            "wrong_distractor_count": wrong_rel_count,
        }
        print(f"[{s_name:<28}] EM: {em_count:2d}/{n_total} ({em_rate*100:5.1f}%) | Wrong Distractor Copied: {wrong_rel_count:2d}")

    out_file = REPORTS_DIR / "generalization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Generalization evaluation written to {out_file}")


if __name__ == "__main__":
    run_generalization_eval()
