"""Runs the Batch 2781 Generation Benchmark on the SFT candidate checkpoint and saves to batch2821_2860."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def compute_bigram_repetition(tokens: List[int]) -> float:
    if len(tokens) < 3:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    unique_bigrams = set(bigrams)
    return 1.0 - (len(unique_bigrams) / len(bigrams))


def run_candidate_benchmark():
    benchmark_dataset_path = REPO_ROOT / "experiments" / "batch2781_2820" / "GENERATION_BENCHMARK.json"
    results_path = REPO_ROOT / "experiments" / "batch2821_2860" / "SFT_CANDIDATE_BATCH2781_BENCHMARK_RESULTS.json"
    cand_ckpt_path = REPO_ROOT / "experiments" / "batch2821_2860" / "sft_pilot_run" / "sft_checkpoint_step_100.pt"
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

    with open(benchmark_dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    tokenizer, _ = load_tokenizer_artifacts(str(tok_dir))
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to("cpu")
    ckpt = torch.load(cand_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    modes = ["raw_completion", "plain_instruction"]
    results_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    print(f"Running candidate on Batch 2781 benchmark (28 cases)...")

    for mode in modes:
        print(f"\n--- Running Mode: {mode} ---")
        for i, c in enumerate(cases, 1):
            cid = c["case_id"]
            cat = c["category"]
            inp = c["input"]
            expected = c["expected_behavior"]

            formatted_p = inp
            if mode == "plain_instruction":
                formatted_p = f"Instruction: {inp.strip()}\nResponse:"

            p_toks = tokenizer.encode(formatted_p)
            input_ids = [0] + p_toks

            curr = list(input_ids)
            t0 = time.perf_counter()
            gen_toks = []
            stop_reason = "MAX_TOKENS"

            for _ in range(32):
                if len(curr) >= 512:
                    stop_reason = "CONTEXT_LIMIT"
                    break
                x = torch.tensor([curr], dtype=torch.long)
                with torch.no_grad():
                    logits = model(x)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                if next_tok == 1:
                    stop_reason = "STOP_TOKEN"
                    break
                gen_toks.append(next_tok)
                curr.append(next_tok)

            latency = (time.perf_counter() - t0) * 1000.0
            out_text = tokenizer.decode(gen_toks)
            bigram_rep = compute_bigram_repetition(gen_toks)

            results_by_mode[mode].append({
                "case_id": cid,
                "category": cat,
                "input": inp,
                "expected_behavior": expected,
                "prompt_mode": mode,
                "generated_tokens": len(gen_toks),
                "latency_ms": round(latency, 2),
                "stop_reason": stop_reason,
                "generated_text": out_text,
                "bigram_repetition_ratio": round(bigram_rep, 4),
            })
            print(f"[{i}/{len(cases)}] {cid} ({cat}): {len(gen_toks)} toks, {latency:.1f}ms, stop={stop_reason}")

    payload = {
        "benchmark_name": "batch2781_benchmark_sft_candidate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(cand_ckpt_path),
        "checkpoint_file_sha256": hashlib.sha256(cand_ckpt_path.read_bytes()).hexdigest(),
        "total_cases": len(cases),
        "results": results_by_mode,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote candidate benchmark results to {results_path}")


if __name__ == "__main__":
    run_candidate_benchmark()
