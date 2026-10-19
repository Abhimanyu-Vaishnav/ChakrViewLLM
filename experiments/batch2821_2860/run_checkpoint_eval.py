"""Evaluates an arbitrary checkpoint on the held-out SFT test set and produces standardized comparison JSON."""

from __future__ import annotations

import argparse
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

logger = logging.getLogger("eval_checkpoint")


def evaluate_checkpoint_on_test_set(checkpoint_path: Path, output_json: Path, name: str):
    eval_set_path = REPO_ROOT / "experiments" / "batch2821_2860" / "SFT_EVALUATION_SET.json"
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    cases: List[Dict[str, Any]] = eval_data["cases"]

    tokenizer, _ = load_tokenizer_artifacts(str(tok_dir))
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to("cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Parameter digest
    hasher = hashlib.sha256()
    for _, p in sorted(model.named_parameters()):
        hasher.update(p.detach().cpu().numpy().tobytes())
    param_digest = hasher.hexdigest()

    file_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    results: List[Dict[str, Any]] = []
    print(f"Evaluating {name} ({checkpoint_path.name}) across {len(cases)} cases...")

    for i, c in enumerate(cases, 1):
        cid = c["id"]
        cat = c["category"]
        prompt = c["prompt"]
        target = c["response"]

        # Tokenize prompt
        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks  # BOS + prompt

        # Greedy generation up to 32 tokens
        curr = list(input_ids)
        t0 = time.perf_counter()
        generated_tokens = []
        stop_reason = "MAX_TOKENS"

        for _ in range(32):
            if len(curr) >= 512:
                stop_reason = "CONTEXT_LIMIT"
                break
            x = torch.tensor([curr], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            if next_tok == 1:  # EOS
                stop_reason = "STOP_TOKEN"
                break
            generated_tokens.append(next_tok)
            curr.append(next_tok)

        latency = (time.perf_counter() - t0) * 1000.0
        out_text = tokenizer.decode(generated_tokens)

        # Automated metrics
        exact_match = target.strip().lower() in out_text.strip().lower()

        json_valid = None
        if cat == "structured_output":
            try:
                json.loads(out_text.strip())
                json_valid = True
            except Exception:
                json_valid = False

        abstention_detected = None
        if cat == "uncertainty_insufficient_info":
            abstention_detected = any(phrase in out_text.lower() for phrase in ["not contain", "unavailable", "does not state", "cannot", "no information"])

        results.append({
            "id": cid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated_text": out_text,
            "generated_tokens": len(generated_tokens),
            "latency_ms": round(latency, 2),
            "stop_reason": stop_reason,
            "target_exact_match": exact_match,
            "json_valid": json_valid,
            "abstention_detected": abstention_detected,
        })
        print(f"[{i}/{len(cases)}] {cid} ({cat}): match={exact_match}, json={json_valid}, abst={abstention_detected}")

    payload = {
        "evaluation_name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": file_sha,
        "parameter_digest": param_digest,
        "total_cases": len(cases),
        "results": results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote results to {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--name", type=str, default="evaluation")
    args = parser.parse_args()

    evaluate_checkpoint_on_test_set(
        checkpoint_path=Path(args.checkpoint),
        output_json=Path(args.output),
        name=args.name,
    )


if __name__ == "__main__":
    main()
