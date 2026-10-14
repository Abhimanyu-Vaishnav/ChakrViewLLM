"""
Batches 271-280: Fresh Randomized Generalization Benchmark Runner.
Location: experiments/batch271_280_compound_reasoning/run_fresh_benchmark.py

Compares evaluation arms on the fresh 200-question generalization manifest:
Arm A: Neural Core Only (canonical 3.44M ChakrMicro greedy decoding)
Arm B: Neural Core + Pointer Head (learned relational binding)
Arm C: Deterministic / Tool-Assisted Runtime (Calculator + Compound Arithmetic Engine)
Arm D: Retrieval-Assisted Runtime (Knowledge / Multi-hop retriever)
Arm E: Full Integrated Runtime (Founder Console Entrypoint / ChakrViewRuntimeSession)

Uses RobustAnswerScorer (Batch 273) to ensure:
- Exact numeric boundary checking (no false substring matches like 12 in 120)
- Sandboxed code validation and safe syntax/execution checks
- Balanced epistemic precision/recall for unanswerable / false-premise questions
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.robust_scoring import RobustAnswerScorer
from scripts.founder_manual_test import ChakrViewRuntimeSession

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_DIR = Path("experiments/batch271_280_compound_reasoning/data")
REPORTS_DIR = Path("experiments/batch271_280_compound_reasoning/reports")


def run_neural_core_arm(model, tokenizer, test_items: List[Dict[str, Any]], device=torch.device("cpu")) -> Dict[str, Any]:
    t0 = time.perf_counter()
    correct = 0
    total = len(test_items)
    results = []

    for item in test_items:
        prompt = f"Question: {item['query']}\nAnswer:"
        input_ids = torch.tensor([[0] + tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        # Greedy generation up to 16 tokens
        generated = list(input_ids[0].tolist())
        with torch.no_grad():
            for _ in range(16):
                x = torch.tensor([generated[-128:]], dtype=torch.long, device=device)
                logits = model(x)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                if next_tok in (1, 2):  # EOS / EOD
                    break
                generated.append(next_tok)

        ans_tokens = generated[len(input_ids[0]):]
        ans_text = tokenizer.decode(ans_tokens, errors="replace").strip()

        score_res = RobustAnswerScorer.score(
            query=item["query"],
            generated_answer=ans_text,
            expected_answer=item.get("expected_answer", ""),
            validation_mode=item.get("validation_mode", "contains"),
            is_unanswerable=item.get("is_unanswerable", False),
        )

        if score_res.is_correct:
            correct += 1

        results.append({
            "id": item["id"],
            "query": item["query"],
            "category": item.get("category"),
            "generated": ans_text,
            "expected": item.get("expected_answer"),
            "passed": score_res.is_correct,
            "score_reason": score_res.failure_reason,
            "component_origin": "neural_core_unassisted",
        })

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": "Arm A: Canonical Neural Core (Direct Greedy Decoding)",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_latency_ms": round(latency_ms, 2),
        "component_origin": "neural_core_unassisted",
        "results": results,
    }


def run_full_runtime_arm(session: ChakrViewRuntimeSession, test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    correct = 0
    total = len(test_items)
    category_breakdown: Dict[str, Dict[str, int]] = {}
    origin_breakdown: Dict[str, int] = {}
    results = []

    for item in test_items:
        query = item["query"]
        cat = item.get("category", "unknown")
        if cat not in category_breakdown:
            category_breakdown[cat] = {"total": 0, "correct": 0}
        category_breakdown[cat]["total"] += 1

        resp = session.process_input(query)
        ans_text = (resp.get("final_response") or resp.get("natural_response") or "").strip()
        origin = resp.get("component_origin") or resp.get("strategy", "UNKNOWN")
        origin_breakdown[origin] = origin_breakdown.get(origin, 0) + 1

        score_res = RobustAnswerScorer.score(
            query=query,
            generated_answer=ans_text,
            expected_answer=item.get("expected_answer", ""),
            validation_mode=item.get("validation_mode", "contains"),
            is_unanswerable=item.get("is_unanswerable", False),
        )

        if score_res.is_correct:
            correct += 1
            category_breakdown[cat]["correct"] += 1

        results.append({
            "id": item["id"],
            "query": query,
            "category": cat,
            "intent": resp.get("intent"),
            "component_origin": origin,
            "generated": ans_text,
            "expected": item.get("expected_answer"),
            "passed": score_res.is_correct,
            "score_reason": score_res.failure_reason,
        })

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": "Arm E: Full Integrated Runtime (Founder Console Entrypoint)",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_latency_ms": round(latency_ms, 2),
        "category_breakdown": category_breakdown,
        "origin_breakdown": origin_breakdown,
        "results": results,
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_DIR / "fresh_evaluation_manifest_200.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)

    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"Loaded Fresh Evaluation Manifest: {len(items)} items | SHA-256: {manifest_sha}")

    # Load canonical checkpoint
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    session = ChakrViewRuntimeSession()

    print("\n--- Running Evaluation Arm A: Canonical Neural Core Unassisted ---")
    arm_a = run_neural_core_arm(model, tokenizer, items, device=device)
    print(f"[{arm_a['arm']}] Correct: {arm_a['correct']}/{arm_a['total']} ({arm_a['accuracy']*100:.1f}%) | Latency: {arm_a['avg_latency_ms']}ms")

    print("\n--- Running Evaluation Arm E: Full Integrated Runtime ---")
    arm_e = run_full_runtime_arm(session, items)
    print(f"[{arm_e['arm']}] Correct: {arm_e['correct']}/{arm_e['total']} ({arm_e['accuracy']*100:.1f}%) | Latency: {arm_e['avg_latency_ms']}ms")

    print("\nCategory Breakdown (Full Runtime):")
    for cat, stat in arm_e["category_breakdown"].items():
        pct = (stat["correct"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        print(f"  - {cat:30s}: {stat['correct']}/{stat['total']} ({pct:.1f}%)")

    print("\nComponent Origin Breakdown (Full Runtime):")
    for orig, count in arm_e["origin_breakdown"].items():
        print(f"  - {orig:30s}: {count} queries")

    report_payload = {
        "milestone": "Batches 271-280: Fresh 200-Item Generalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "total_items": len(items),
        "scoring_protocol_version": "v2.0_robust_bounded",
        "arms": {
            "Arm_A_neural_core_unassisted": {
                "name": arm_a["arm"],
                "total": arm_a["total"],
                "correct": arm_a["correct"],
                "accuracy": arm_a["accuracy"],
                "avg_latency_ms": arm_a["avg_latency_ms"],
            },
            "Arm_E_full_runtime_assisted": {
                "name": arm_e["arm"],
                "total": arm_e["total"],
                "correct": arm_e["correct"],
                "accuracy": arm_e["accuracy"],
                "avg_latency_ms": arm_e["avg_latency_ms"],
                "category_breakdown": arm_e["category_breakdown"],
                "origin_breakdown": arm_e["origin_breakdown"],
            }
        },
        "sample_item_evaluations": arm_e["results"][:30],
    }

    out_file = REPORTS_DIR / "fresh_generalization_results.json"
    out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nSaved fresh generalization benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
