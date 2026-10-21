"""Audited Evaluation Engine for ChakrView v0.8 Single-Task JSON SFT Experiment.

Evaluates checkpoints on the frozen test set (experiments/research_v08_json_sft/data/test.jsonl):
1. 20 JSON Extraction tasks:
   - JSON parseability (json.loads)
   - Exact schema and type match (dict with keys {"status", "code"}, str and int)
   - Field correctness: status_match, code_match
   - Full semantic task success (schema match AND status_match AND code_match)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

logger = logging.getLogger("v08_json_evaluator")

EVALUATOR_VERSION = "v0.8.0-json-strict"


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
    expected_code: Optional[int] = None,
    repetition_ratio: float = 0.0,
    is_empty: bool = False,
    first_token_is_eos: bool = False,
) -> Dict[str, Any]:
    """Pure evaluation logic separated from model inference for deterministic auditing."""
    if category == "two_key_json_extraction":
        json_parsed = False
        parsed_obj = None
        schema_valid = False
        status_match = False
        code_match = False
        full_semantic_success = False

        if not is_empty and not first_token_is_eos:
            # 1. Parseability
            clean_text = output_text.strip()
            # Handle potential markdown code fencing if present
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

            # 2. Schema correctness
            if json_parsed and isinstance(parsed_obj, dict):
                keys = set(parsed_obj.keys())
                if keys == {"status", "code"}:
                    status_val = parsed_obj["status"]
                    code_val = parsed_obj["code"]
                    if isinstance(status_val, str) and isinstance(code_val, int) and not isinstance(code_val, bool):
                        schema_valid = True

            # 3. Field and semantic correctness
            if schema_valid:
                assert parsed_obj is not None
                if expected_status is not None and parsed_obj.get("status") == expected_status:
                    status_match = True
                if expected_code is not None and parsed_obj.get("code") == expected_code:
                    code_match = True
                if status_match and code_match:
                    full_semantic_success = True

        return {
            "json_parsed": json_parsed,
            "schema_valid": schema_valid,
            "status_match": status_match,
            "code_match": code_match,
            "full_semantic_success": full_semantic_success,
            "replay_non_empty": None,
            "replay_meaningful": None,
        }

    elif category == "base_continuation_replay":
        replay_non_empty = not is_empty and not first_token_is_eos
        replay_meaningful = replay_non_empty and (repetition_ratio < 0.40)
        return {
            "json_parsed": None,
            "schema_valid": None,
            "status_match": None,
            "code_match": None,
            "full_semantic_success": None,
            "replay_non_empty": replay_non_empty,
            "replay_meaningful": replay_meaningful,
        }

    else:
        raise ValueError(f"Unknown evaluation category: {category}")


def evaluate_checkpoint_on_test_split(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    test_jsonl_path: Path,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    ckpt_sha = compute_sha256(checkpoint_path)
    tok_vocab_sha = compute_sha256(tokenizer_dir / "vocab.json")
    tok_merges_sha = compute_sha256(tokenizer_dir / "merges.json")
    test_file_sha = compute_sha256(test_jsonl_path)

    tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = ChakrMicro(ModelConfig(max_seq_len=256))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    json_cases = [r for r in records if r["category"] == "two_key_json_extraction"]
    replay_cases = [r for r in records if r["category"] == "base_continuation_replay"]

    per_example_results = []
    
    # Aggregates
    json_parsed_count = 0
    schema_valid_count = 0
    status_match_count = 0
    code_match_count = 0
    full_semantic_success_count = 0
    json_empty_count = 0
    json_immediate_eos_count = 0
    json_truncated_count = 0

    replay_non_empty_count = 0
    replay_meaningful_count = 0
    replay_empty_count = 0
    replay_immediate_eos_count = 0

    for r in records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        target = r["response"]

        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks  # Prepend BOS

        gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=max_new_tokens)

        eval_res = evaluate_output_case(
            category=cat,
            output_text=gen["text"],
            expected_status=r.get("expected_status"),
            expected_code=r.get("expected_code"),
            repetition_ratio=gen["repetition_ratio"],
            is_empty=gen["is_empty"],
            first_token_is_eos=gen["first_token_is_eos"],
        )

        if cat == "two_key_json_extraction":
            if eval_res["json_parsed"]:
                json_parsed_count += 1
            if eval_res["schema_valid"]:
                schema_valid_count += 1
            if eval_res["status_match"]:
                status_match_count += 1
            if eval_res["code_match"]:
                code_match_count += 1
            if eval_res["full_semantic_success"]:
                full_semantic_success_count += 1
            if gen["is_empty"]:
                json_empty_count += 1
            if gen["first_token_is_eos"]:
                json_immediate_eos_count += 1
            if gen["stop_reason"] == "MAX_TOKENS":
                json_truncated_count += 1

        elif cat == "base_continuation_replay":
            if eval_res["replay_non_empty"]:
                replay_non_empty_count += 1
            if eval_res["replay_meaningful"]:
                replay_meaningful_count += 1
            if gen["is_empty"]:
                replay_empty_count += 1
            if gen["first_token_is_eos"]:
                replay_immediate_eos_count += 1

        per_example_results.append({
            "id": rid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated_text": gen["text"],
            "generated_tokens": gen["token_count"],
            "first_token_id": gen["first_token_id"],
            "first_token_is_eos": gen["first_token_is_eos"],
            "stop_reason": gen["stop_reason"],
            "is_empty": gen["is_empty"],
            "repetition_ratio": gen["repetition_ratio"],
            "latency_ms": gen["latency_ms"],
            **eval_res,
        })

    total_json = len(json_cases)
    total_replay = len(replay_cases)

    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": ckpt_sha,
        "tokenizer_vocab_sha256": tok_vocab_sha,
        "tokenizer_merges_sha256": tok_merges_sha,
        "test_dataset_file": test_jsonl_path.name,
        "test_dataset_sha256": test_file_sha,
        "decoding_config": {
            "mode": "greedy_deterministic",
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop_tokens": [1],
        },
        "json_metrics": {
            "total_json_prompts": total_json,
            "json_parsed_numerator": json_parsed_count,
            "json_parsed_denominator": total_json,
            "json_parsed_pct": round(json_parsed_count / total_json * 100.0, 2) if total_json else 0.0,
            "schema_valid_numerator": schema_valid_count,
            "schema_valid_denominator": total_json,
            "schema_valid_pct": round(schema_valid_count / total_json * 100.0, 2) if total_json else 0.0,
            "status_match_numerator": status_match_count,
            "status_match_denominator": total_json,
            "status_match_pct": round(status_match_count / total_json * 100.0, 2) if total_json else 0.0,
            "code_match_numerator": code_match_count,
            "code_match_denominator": total_json,
            "code_match_pct": round(code_match_count / total_json * 100.0, 2) if total_json else 0.0,
            "full_semantic_success_numerator": full_semantic_success_count,
            "full_semantic_success_denominator": total_json,
            "full_semantic_success_pct": round(full_semantic_success_count / total_json * 100.0, 2) if total_json else 0.0,
            "empty_outputs_numerator": json_empty_count,
            "empty_outputs_denominator": total_json,
            "immediate_eos_numerator": json_immediate_eos_count,
            "immediate_eos_denominator": total_json,
            "max_tokens_truncated_numerator": json_truncated_count,
            "max_tokens_truncated_denominator": total_json,
        },
        "replay_metrics": {
            "total_replay_prompts": total_replay,
            "replay_non_empty_numerator": replay_non_empty_count,
            "replay_non_empty_denominator": total_replay,
            "replay_non_empty_pct": round(replay_non_empty_count / total_replay * 100.0, 2) if total_replay else 0.0,
            "replay_meaningful_numerator": replay_meaningful_count,
            "replay_meaningful_denominator": total_replay,
            "replay_meaningful_pct": round(replay_meaningful_count / total_replay * 100.0, 2) if total_replay else 0.0,
            "replay_empty_numerator": replay_empty_count,
            "replay_empty_denominator": total_replay,
            "replay_immediate_eos_numerator": replay_immediate_eos_count,
            "replay_immediate_eos_denominator": total_replay,
        },
    }

    return {
        "summary": summary,
        "per_example_results": per_example_results,
    }
