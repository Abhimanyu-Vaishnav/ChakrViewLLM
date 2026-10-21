"""Evaluation Engine for ChakrView v0.8 Experiment 02: Single-Field String Extraction.

Evaluates checkpoints on the frozen test set (experiments/research_v08_json_sft/exp02_string_copy/data/test.jsonl):
1. 20 Single-Field JSON Extraction tasks:
   - JSON parseability (json.loads)
   - Exact schema and type match (dict with exactly key "status", value is str)
   - Exact copied-string accuracy (model output matches expected status string bit-exact)
   - Degenerate outputs: is_empty, first_token_is_eos, repetition_ratio, max_tokens truncation
2. 10 Continuation Replay tasks:
   - Non-empty output
   - Non-degenerate continuation (repetition_ratio < 0.40)
   - Token count & first token

Deterministic Greedy Decoding:
- T = 0.0 (argmax)
- max_new_tokens = 64
- Stop token = [1] (<EOS>)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

logger = logging.getLogger("exp02_evaluator")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bigram_repetition(tokens: List[int]) -> float:
    if len(tokens) < 3:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def generate_greedy(
    model: ChakrMicro,
    tokenizer: Any,
    input_tokens: List[int],
    max_new_tokens: int = 64,
    context_limit: int = 256,
) -> Dict[str, Any]:
    curr = list(input_tokens)
    t0 = time.perf_counter()
    gen_tokens = []
    stop_reason = "MAX_TOKENS"
    first_token_id = None
    first_token_is_eos = False

    for step_idx in range(max_new_tokens):
        if len(curr) >= context_limit:
            stop_reason = "CONTEXT_LIMIT"
            break
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if step_idx == 0:
            first_token_id = next_tok
            first_token_is_eos = (next_tok == 1)

        if next_tok == 1:  # EOS_ID
            stop_reason = "STOP_TOKEN"
            break
        gen_tokens.append(next_tok)
        curr.append(next_tok)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        out_text = tokenizer.decode(gen_tokens, errors="replace")
    except Exception:
        out_text = tokenizer.decode(gen_tokens)

    return {
        "generated_tokens": gen_tokens,
        "token_count": len(gen_tokens),
        "first_token_id": first_token_id,
        "first_token_is_eos": first_token_is_eos,
        "text": out_text,
        "latency_ms": round(latency_ms, 2),
        "stop_reason": stop_reason,
        "is_empty": len(gen_tokens) == 0,
        "repetition_ratio": round(compute_bigram_repetition(gen_tokens), 4),
    }


def evaluate_output_case(
    category: str,
    output_text: str,
    expected_status: Optional[str] = None,
    repetition_ratio: float = 0.0,
    is_empty: bool = False,
    first_token_is_eos: bool = False,
) -> Dict[str, Any]:
    """Pure evaluation logic separated from model inference for deterministic auditing."""
    if category == "single_field_string_extraction":
        json_parsed = False
        parsed_obj = None
        schema_valid = False
        status_match = False
        full_semantic_success = False

        if not is_empty and not first_token_is_eos:
            clean_text = output_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            try:
                parsed_obj = json.loads(clean_text)
                json_parsed = True
            except Exception:
                json_parsed = False

            if json_parsed and isinstance(parsed_obj, dict):
                keys = set(parsed_obj.keys())
                if keys == {"status"} and isinstance(parsed_obj.get("status"), str):
                    schema_valid = True

            if schema_valid and expected_status is not None:
                status_match = (parsed_obj.get("status") == expected_status)
                full_semantic_success = status_match

        return {
            "json_parsed": json_parsed,
            "schema_valid": schema_valid,
            "status_match": status_match,
            "full_semantic_success": full_semantic_success,
            "is_empty": is_empty,
            "first_token_is_eos": first_token_is_eos,
            "repetition_ratio": repetition_ratio,
            "parsed_output": parsed_obj,
        }

    elif category == "base_continuation_replay":
        return {
            "non_empty": not is_empty,
            "meaningful_continuation": not is_empty and repetition_ratio < 0.40,
            "first_token_is_eos": first_token_is_eos,
            "repetition_ratio": repetition_ratio,
        }

    return {}


def run_evaluation(checkpoint_path: Path, output_json_path: Path, role_name: str) -> Dict[str, Any]:
    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    test_jsonl = Path(__file__).resolve().parent / "data" / "test.jsonl"

    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    test_sha = compute_sha256(test_jsonl)
    ckpt_sha = compute_sha256(checkpoint_path)

    print(f"\n[Evaluating {role_name}]")
    print(f"  Checkpoint: {checkpoint_path.name} (SHA: {ckpt_sha[:16]}...)")
    print(f"  Test File:  {test_jsonl.name} (SHA: {test_sha[:16]}...)")

    device = torch.device("cpu")
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in raw_state:
        model.load_state_dict(raw_state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(raw_state, strict=True)
    model.eval()

    records = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    eval_results = []
    json_metrics = {
        "total": 0,
        "json_parsed": 0,
        "schema_valid": 0,
        "status_match": 0,
        "full_semantic_success": 0,
        "empty_outputs": 0,
        "immediate_eos": 0,
        "max_tokens_truncations": 0,
    }
    replay_metrics = {
        "total": 0,
        "non_empty": 0,
        "meaningful_continuation": 0,
        "immediate_eos": 0,
    }

    for idx, rec in enumerate(records):
        cat = rec.get("category")
        prompt = rec["prompt"]
        prompt_str = prompt if prompt.endswith("\n") else prompt.strip()
        p_toks = tokenizer.encode(prompt_str)
        input_seq = [0] + p_toks

        gen_out = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            input_tokens=input_seq,
            max_new_tokens=64,
            context_limit=256,
        )

        case_res = evaluate_output_case(
            category=cat,
            output_text=gen_out["text"],
            expected_status=rec.get("expected_status"),
            repetition_ratio=gen_out["repetition_ratio"],
            is_empty=gen_out["is_empty"],
            first_token_is_eos=gen_out["first_token_is_eos"],
        )

        item_result = {
            "record_id": rec["id"],
            "category": cat,
            "prompt": prompt,
            "expected_response": rec.get("response"),
            "expected_status": rec.get("expected_status"),
            "raw_output": gen_out["text"],
            "generated_tokens": gen_out["generated_tokens"],
            "token_count": gen_out["token_count"],
            "latency_ms": gen_out["latency_ms"],
            "stop_reason": gen_out["stop_reason"],
            "evaluation": case_res,
        }
        eval_results.append(item_result)

        if cat == "single_field_string_extraction":
            json_metrics["total"] += 1
            if case_res["json_parsed"]:
                json_metrics["json_parsed"] += 1
            if case_res["schema_valid"]:
                json_metrics["schema_valid"] += 1
            if case_res["status_match"]:
                json_metrics["status_match"] += 1
            if case_res["full_semantic_success"]:
                json_metrics["full_semantic_success"] += 1
            if case_res["is_empty"]:
                json_metrics["empty_outputs"] += 1
            if case_res["first_token_is_eos"]:
                json_metrics["immediate_eos"] += 1
            if gen_out["stop_reason"] == "MAX_TOKENS":
                json_metrics["max_tokens_truncations"] += 1

        elif cat == "base_continuation_replay":
            replay_metrics["total"] += 1
            if case_res["non_empty"]:
                replay_metrics["non_empty"] += 1
            if case_res["meaningful_continuation"]:
                replay_metrics["meaningful_continuation"] += 1
            if case_res["first_token_is_eos"]:
                replay_metrics["immediate_eos"] += 1

    summary = {
        "role": role_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha,
        "test_dataset_sha256": test_sha,
        "json_extraction_metrics": json_metrics,
        "replay_metrics": replay_metrics,
        "results": eval_results,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  [Result Summary for {role_name}]:")
    print(f"    JSON Parse Rate:         {json_metrics['json_parsed']}/{json_metrics['total']} ({json_metrics['json_parsed']/max(1, json_metrics['total'])*100:.1f}%)")
    print(f"    Schema Valid Rate:       {json_metrics['schema_valid']}/{json_metrics['total']} ({json_metrics['schema_valid']/max(1, json_metrics['total'])*100:.1f}%)")
    print(f"    Status Match (Copying):  {json_metrics['status_match']}/{json_metrics['total']} ({json_metrics['status_match']/max(1, json_metrics['total'])*100:.1f}%)")
    print(f"    Full Semantic Success:   {json_metrics['full_semantic_success']}/{json_metrics['total']} ({json_metrics['full_semantic_success']/max(1, json_metrics['total'])*100:.1f}%)")
    print(f"    Replay Non-Empty Rate:   {replay_metrics['non_empty']}/{replay_metrics['total']}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate_exp02.py <checkpoint_path> <output_json> <role_name>")
        sys.exit(1)
    run_evaluation(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
