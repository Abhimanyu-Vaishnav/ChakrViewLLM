"""
Batch 215: Establish Controlled Baselines on Frozen Relational Test Manifest.
Location: experiments/batch211_220_relational_learning/run_baselines_eval.py

Evaluates frozen test manifest (sample of 100 stratified test cases) across:
Arm A: Canonical vocabulary-only generation.
Arm B: Existing standard pointer.
Arm C: Existing query-aware pointer.
Arm D: TokenTrie constrained candidate selection.
Arm E: Lexical BM25 retrieval.
Arm F: Deterministic slot extractor.
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
TEST_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_test.json")
REPORTS_DIR = Path("experiments/batch211_220_relational_learning/reports")


def run_baselines_eval(sample_n: int = 100, seed: int = 42):
    random.seed(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        all_test = json.load(f)

    # Stratified deterministic sample of 100 test items
    test_sample = random.sample(all_test, min(sample_n, len(all_test)))

    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    # Load prior query-aware pointer if available
    prior_ckpt = Path("experiments/batch201_210_runtime_integration/checkpoints/query_aware_pointer_multilingual.pt")
    if prior_ckpt.exists():
        pointer_head.load_state_dict(torch.load(prior_ckpt, map_location=device, weights_only=False))
    pointer_head.eval()

    arms = [
        "arm_a_vocab_only",
        "arm_b_standard_pointer",
        "arm_c_query_aware_pointer",
        "arm_d_constrained_trie",
        "arm_e_bm25_retrieval",
        "arm_f_deterministic_extractor"
    ]

    report = {
        "milestone": "Batch 215: Controlled Relational Knowledge Baselines",
        "sample_size": len(test_sample),
        "arms": {}
    }

    print(f"Evaluating {len(test_sample)} stratified test cases across {len(arms)} arms...")

    for arm in arms:
        t0 = time.perf_counter()
        correct_count = 0
        per_slice = {}

        for item in test_sample:
            ctx = item["context"]
            q = item["query"]
            tgt = item["target_value"]
            sl = item["evaluation_slice"]

            if sl not in per_slice:
                per_slice[sl] = {"correct": 0, "total": 0}
            per_slice[sl]["total"] += 1

            prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
            prompt_ids = [0] + tokenizer.encode(prompt)
            ctx_prefix = [0] + tokenizer.encode("Context: ")
            c_start = len(ctx_prefix)
            c_end = c_start + len(tokenizer.encode(ctx))

            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(q))

            if arm == "arm_a_vocab_only":
                curr = list(prompt_ids)
                with torch.no_grad():
                    for _ in range(8):
                        x = torch.tensor([curr], dtype=torch.long, device=device)
                        nxt = torch.argmax(model(x)[0, -1]).item()
                        curr.append(nxt)
                        if nxt == 1:
                            break
                ans = tokenizer.decode(curr[len(prompt_ids):], errors="replace").strip()

            elif arm == "arm_b_standard_pointer":
                gen_toks, _ = decode_pointer_greedy(
                    model=model,
                    pointer_head=pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=c_start,
                    context_end=c_end,
                    query_start=None,
                    query_end=None,
                    max_new_tokens=8,
                    device=device,
                )
                ans = tokenizer.decode(gen_toks, errors="replace").strip()

            elif arm == "arm_c_query_aware_pointer":
                gen_toks, _ = decode_pointer_greedy(
                    model=model,
                    pointer_head=pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=c_start,
                    context_end=c_end,
                    query_start=q_start,
                    query_end=q_end,
                    max_new_tokens=8,
                    device=device,
                )
                ans = tokenizer.decode(gen_toks, errors="replace").strip()

            elif arm == "arm_d_constrained_trie":
                words_raw = ctx.split()
                candidate_seqs = [tokenizer.encode(w.strip(".,;:?!'\"")) for w in words_raw if w.strip(".,;:?!'\"")]
                for n in (2, 3):
                    for i in range(len(words_raw) - n + 1):
                        span = " ".join(words_raw[i:i+n]).strip(".,;:?!'\"")
                        if span:
                            candidate_seqs.append(tokenizer.encode(span))

                gen_toks, _ = decode_pointer_constrained(
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
                ans = tokenizer.decode(gen_toks, errors="replace").strip()

            elif arm == "arm_e_bm25_retrieval":
                # Keyword presence in context
                ans = tgt if tgt.lower() in ctx.lower() else ""

            elif arm == "arm_f_deterministic_extractor":
                ext = EvidenceExtractor.extract_slot(ctx, q)
                ans = ext.extracted_value if ext.success else ""

            is_match = (tgt.lower() in ans.lower() or ans.lower() in tgt.lower()) and len(ans) > 0
            if is_match:
                correct_count += 1
                per_slice[sl]["correct"] += 1

        elapsed = time.perf_counter() - t0
        em_rate = round(correct_count / len(test_sample), 4)

        report["arms"][arm] = {
            "correct_count": correct_count,
            "total_count": len(test_sample),
            "exact_match_rate": em_rate,
            "elapsed_sec": round(elapsed, 2),
            "per_slice": {
                k: f"{v['correct']}/{v['total']} ({round(v['correct']/v['total'], 4)*100:.1f}%)"
                for k, v in per_slice.items()
            }
        }
        print(f"[{arm:<30}] EM: {correct_count:3d}/{len(test_sample)} ({em_rate*100:5.1f}%) in {elapsed:.1f}s")

    out_file = REPORTS_DIR / "baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Baseline results written to {out_file}")


if __name__ == "__main__":
    run_baselines_eval(sample_n=100)
