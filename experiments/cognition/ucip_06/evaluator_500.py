"""500-Probe Independent Benchmark Evaluator for UCIP-06.

Strict, multi-metric scoring across 10 distinct language & competence categories.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.tokenizer import EOS_ID
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
BENCHMARK_PATH = REPO_ROOT / "experiments" / "cognition" / "ucip_06" / "sealed_500_benchmark.json"


def evaluate_probe(
    category: str,
    prompt: str,
    output: str,
    expected_keywords: List[str],
    rubric: str,
    exact_match: bool,
) -> Dict[str, Any]:
    """Score a single probe based on strict, deterministic rubrics."""
    cleaned = output.strip()
    norm_out = cleaned.lower()

    # Base failure check
    is_empty = len(cleaned) == 0
    words = cleaned.split()
    repetitive = len(words) > 4 and len(set(words)) <= len(words) // 3

    success = False
    details = ""

    if is_empty:
        return {"success": False, "details": "Empty generation", "is_empty": True, "is_repetitive": False}

    if repetitive:
        return {"success": False, "details": "Repetitive loop detected", "is_empty": False, "is_repetitive": True}

    if exact_match or category == "Exact Copying":
        # Target must be exact substring in output
        target = expected_keywords[0] if expected_keywords else ""
        norm_target = target.lower().strip()
        if norm_target in norm_out:
            success = True
            details = f"Exact substring match for '{norm_target}'"
        else:
            details = f"Expected '{norm_target}', got '{norm_out}'"

    elif category == "Elementary Arithmetic":
        # Check if any expected keyword (e.g. '3', 'three') appears as an isolated token or word
        matched_kw = []
        for kw in expected_keywords:
            kw_clean = kw.lower().strip()
            # Match word boundary or standalone number
            pattern = rf"(?:\b|_){re.escape(kw_clean)}(?:\b|_)"
            if re.search(pattern, norm_out) or kw_clean in norm_out:
                matched_kw.append(kw_clean)
        if matched_kw:
            success = True
            details = f"Found arithmetic answer: {matched_kw}"
        else:
            details = f"Expected one of {expected_keywords}, got '{norm_out}'"

    elif category == "Devanagari Hindi":
        # Must contain Devanagari characters and at least one expected keyword (if keywords provided)
        dev_chars = len(re.findall(r"[\u0900-\u097F]", cleaned))
        if dev_chars >= 3:
            kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
            if kw_hits or not expected_keywords:
                success = True
                details = f"Devanagari ({dev_chars} chars), kw_hits={kw_hits}"
            else:
                details = f"Devanagari ({dev_chars} chars), but missing keywords {expected_keywords}"
        else:
            details = f"Insufficient Devanagari chars: {dev_chars}"

    elif category in ("Roman Hindi", "Hinglish"):
        kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
        if len(kw_hits) >= 1:
            success = True
            details = f"Keywords hit: {kw_hits}"
        else:
            details = f"Missing Roman/Hinglish keywords {expected_keywords}"

    elif category == "Basic Instructions":
        kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
        if len(kw_hits) >= 1:
            success = True
            details = f"Constraint keywords hit: {kw_hits}"
        else:
            details = f"Instruction unfulfilled, expected {expected_keywords}"

    elif category == "Semantic Consistency":
        kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
        if len(kw_hits) >= 1:
            success = True
            details = f"Consistent keywords hit: {kw_hits}"
        else:
            details = f"Semantic drift, expected {expected_keywords}"

    else:
        # English Conversation, English Simple Explanations, English Sentence Generation
        kw_hits = [kw for kw in expected_keywords if kw.lower() in norm_out]
        if len(kw_hits) >= 1:
            success = True
            details = f"Keywords hit: {kw_hits}"
        else:
            details = f"Missing topic keywords {expected_keywords}"

    return {
        "success": success,
        "details": details,
        "is_empty": is_empty,
        "is_repetitive": repetitive,
    }

    return {
        "success": success,
        "details": details,
        "is_empty": is_empty,
        "is_repetitive": repetitive,
    }


def run_benchmark(
    checkpoint_path: Path,
    output_report_path: Optional[Path] = None,
    device_name: str = "cpu",
    max_tokens: int = 32,
    temperature: float = 0.3,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Load model and run full 500-probe benchmark."""
    device = torch.device(device_name)
    tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw_config = checkpoint.get("config", {})
    if isinstance(raw_config, dict):
        model_config = ModelConfig(**raw_config)
    else:
        model_config = raw_config

    model = ChakrMicro(model_config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)

    categories = sorted(list(set(p["category"] for p in probes)))
    print(f"Evaluating {len(probes)} probes across {len(categories)} categories...")

    category_results: Dict[str, Dict[str, int]] = {}
    for cat in categories:
        category_results[cat] = {"total": 0, "passed": 0, "empty": 0, "repetitive": 0}

    detailed_evals = []

    for i, probe in enumerate(probes):
        pid = probe["probe_id"]
        category = probe["category"]
        prompt = probe["input"]
        expected_keywords = probe.get("expected_keywords", [])
        rubric = probe.get("rubric", "")
        exact_match = probe.get("exact_match", False)

        # Format prompt according to ChakrView standard: "User: <prompt>\nAssistant:"
        formatted_prompt = f"User: {prompt}\nAssistant:"
        input_ids = tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_tokens = []
            curr_tensor = input_tensor

            for _ in range(max_tokens):
                logits = model(curr_tensor)
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

                if top_k > 0:
                    val, idx = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    probs = torch.softmax(val, dim=-1)
                    chosen_idx = torch.multinomial(probs, num_samples=1)
                    next_token = idx[chosen_idx].item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token == EOS_ID:
                    break
                output_tokens.append(next_token)
                curr_tensor = torch.cat([curr_tensor, torch.tensor([[next_token]], device=device)], dim=1)
                if curr_tensor.shape[1] >= model_config.max_seq_len:
                    break

        generated_text = tokenizer.decode(output_tokens, errors="replace").strip()

        eval_res = evaluate_probe(category, prompt, generated_text, expected_keywords, rubric, exact_match)

        category_results[category]["total"] += 1
        if eval_res["success"]:
            category_results[category]["passed"] += 1
        if eval_res["is_empty"]:
            category_results[category]["empty"] += 1
        if eval_res["is_repetitive"]:
            category_results[category]["repetitive"] += 1

        detailed_evals.append({
            "id": pid,
            "category": category,
            "prompt": prompt,
            "expected_keywords": expected_keywords,
            "generated": generated_text,
            "passed": eval_res["success"],
            "details": eval_res["details"],
        })

        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(probes)} probes...")

    total_probes = len(probes)
    total_passed = sum(cat_data["passed"] for cat_data in category_results.values())
    overall_acc = (total_passed / total_probes) * 100.0 if total_probes else 0.0

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_probes": total_probes,
        "total_passed": total_passed,
        "overall_accuracy_percent": round(overall_acc, 2),
        "category_results": category_results,
    }

    print("\n--- Benchmark Summary ---")
    print(f"Overall Accuracy: {total_passed}/{total_probes} ({overall_acc:.2f}%)")
    for cat, stats in category_results.items():
        cat_acc = (stats["passed"] / stats["total"]) * 100.0 if stats["total"] else 0.0
        print(f"  {cat:<30}: {stats['passed']:>3}/{stats['total']:>3} ({cat_acc:6.2f}%) [Rep: {stats['repetitive']}]")

    if output_report_path:
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "evaluations": detailed_evals}, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation report to: {output_report_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 500-probe benchmark on a ChakrView checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(Path(args.checkpoint), Path(args.out) if args.out else None)
