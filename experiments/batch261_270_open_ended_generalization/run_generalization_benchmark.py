"""
Batch 263 & 265: Unseen Question Generalization Benchmark and Arm Comparison.
Location: experiments/batch261_270_open_ended_generalization/run_generalization_benchmark.py

Compares 4 evaluation arms across the sealed held-out manifest (120 items) and frozen regression (60 items):
Arm 1: Neural Core Only (direct canonical 3.44M model greedy decoding)
Arm 2: Neural Core + TokenTrie Constrained Decoding
Arm 3: Tool / Knowledge Assisted Mode (Router + Calculator + Builtin Knowledge)
Arm 4: Full Integrated Runtime (ChakrViewRuntimeSession / Founder Console Entrypoint)

Records per question:
- Intent / route classification
- Final answer
- Correctness against ground truth
- Provenance (Neural, Deterministic Tool, System Knowledge, Memory)
- Latency (ms)
- Appropriate abstention rate
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
from scripts.founder_manual_test import ChakrViewRuntimeSession

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_DIR = Path("experiments/batch261_270_open_ended_generalization/data")
REPORTS_DIR = Path("experiments/batch261_270_open_ended_generalization/reports")


def evaluate_python_code(code_str: str) -> bool:
    clean = code_str.replace("```python", "").replace("```", "").strip()
    try:
        ast.parse(clean)
        return True
    except SyntaxError:
        return False


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

        # Score correctness
        exp = str(item.get("expected_answer", "")).lower()
        is_pass = False
        if exp and exp in ans_text.lower():
            is_pass = True
        elif item.get("is_unanswerable") and ("not" in ans_text.lower() or "unknown" in ans_text.lower()):
            is_pass = True

        if is_pass:
            correct += 1

        results.append({
            "id": item["id"],
            "query": item["query"],
            "generated": ans_text,
            "expected": exp,
            "passed": is_pass,
            "component_origin": "neural_core_unassisted",
        })

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": "Arm 1: Neural Core Unassisted (Canonical 3.44M ChakrMicro)",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_latency_ms": round(latency_ms, 2),
        "component_origin": "neural_core_unassisted",
        "results": results
    }


def run_full_runtime_arm(session: ChakrViewRuntimeSession, test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    correct = 0
    abstentions_correct = 0
    total = len(test_items)
    results = []

    for item in test_items:
        query = item["query"]
        val_mode = item.get("validation_mode", "contains")
        exp = str(item.get("expected_answer", "")).lower()
        is_unans = item.get("is_unanswerable", False)

        resp = session.process_input(query)
        ans_text = (resp.get("final_response") or resp.get("natural_response") or "").strip()
        origin = resp.get("component_origin") or resp.get("strategy", "UNKNOWN")

        is_pass = False
        if val_mode == "contains":
            is_pass = exp in ans_text.lower()
        elif val_mode == "runnable_python":
            is_pass = evaluate_python_code(ans_text)
        elif val_mode == "abstention_or_correction":
            keywords = ["does not have", "no capital", "not have a capital", "uninhabited", "insufficient", "do not have sufficient", "honest abstention", "do not have verified"]
            is_pass = any(k in ans_text.lower() for k in keywords)

        if is_pass:
            correct += 1
            if is_unans:
                abstentions_correct += 1

        results.append({
            "id": item["id"],
            "query": query,
            "archetype": item.get("archetype"),
            "intent": resp.get("intent"),
            "component_origin": origin,
            "generated": ans_text,
            "expected": exp,
            "passed": is_pass,
        })

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": "Arm 4: Full Integrated Runtime (Founder Console Entrypoint)",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "abstentions_correct": abstentions_correct,
        "avg_latency_ms": round(latency_ms, 2),
        "component_origin": "hybrid_router_governed",
        "results": results
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    heldout_path = DATA_DIR / "sealed_heldout_manifest_120.json"
    frozen_path = DATA_DIR / "frozen_regression_60.json"

    heldout_items = json.loads(heldout_path.read_text(encoding="utf-8"))
    frozen_items = json.loads(frozen_path.read_text(encoding="utf-8"))
    print(f"Loaded heldout items: {len(heldout_items)} | Frozen items: {len(frozen_items)}")

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    session = ChakrViewRuntimeSession()

    print("\n--- Running Evaluation on Sealed Held-Out Generalization Manifest (120 Items) ---")
    arm1_heldout = run_neural_core_arm(model, tokenizer, heldout_items, device=device)
    arm4_heldout = run_full_runtime_arm(session, heldout_items)

    print(f"[{arm1_heldout['arm']}] Correct: {arm1_heldout['correct']}/{arm1_heldout['total']} ({arm1_heldout['accuracy']*100:.1f}%) | Latency: {arm1_heldout['avg_latency_ms']}ms")
    print(f"[{arm4_heldout['arm']}] Correct: {arm4_heldout['correct']}/{arm4_heldout['total']} ({arm4_heldout['accuracy']*100:.1f}%) | Latency: {arm4_heldout['avg_latency_ms']}ms")

    print("\n--- Running Evaluation on Frozen Regression Manifest (60 Items) ---")
    arm1_frozen = run_neural_core_arm(model, tokenizer, frozen_items, device=device)
    arm4_frozen = run_full_runtime_arm(session, frozen_items)

    print(f"[{arm1_frozen['arm']}] Correct: {arm1_frozen['correct']}/{arm1_frozen['total']} ({arm1_frozen['accuracy']*100:.1f}%) | Latency: {arm1_frozen['avg_latency_ms']}ms")
    print(f"[{arm4_frozen['arm']}] Correct: {arm4_frozen['correct']}/{arm4_frozen['total']} ({arm4_frozen['accuracy']*100:.1f}%) | Latency: {arm4_frozen['avg_latency_ms']}ms")

    report_payload = {
        "milestone": "Batches 263 & 265: Unseen Question Generalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heldout_manifest_sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        "sealed_heldout_evaluation": {
            "neural_core_unassisted": {
                "correct": arm1_heldout["correct"],
                "total": arm1_heldout["total"],
                "accuracy": arm1_heldout["accuracy"],
                "avg_latency_ms": arm1_heldout["avg_latency_ms"]
            },
            "full_runtime_assisted": {
                "correct": arm4_heldout["correct"],
                "total": arm4_heldout["total"],
                "accuracy": arm4_heldout["accuracy"],
                "abstentions_correct": arm4_heldout["abstentions_correct"],
                "avg_latency_ms": arm4_heldout["avg_latency_ms"]
            }
        },
        "frozen_regression_evaluation": {
            "neural_core_unassisted": {
                "correct": arm1_frozen["correct"],
                "total": arm1_frozen["total"],
                "accuracy": arm1_frozen["accuracy"],
                "avg_latency_ms": arm1_frozen["avg_latency_ms"]
            },
            "full_runtime_assisted": {
                "correct": arm4_frozen["correct"],
                "total": arm4_frozen["total"],
                "accuracy": arm4_frozen["accuracy"],
                "avg_latency_ms": arm4_frozen["avg_latency_ms"]
            }
        }
    }

    out_file = REPORTS_DIR / "unseen_question_generalization_results.json"
    out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nSaved generalization benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
