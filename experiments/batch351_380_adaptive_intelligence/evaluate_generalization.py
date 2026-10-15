"""
Batches 371-375: Generalization Evaluator (Neural-Only vs Integrated Runtime).
Location: experiments/batch351_380_adaptive_intelligence/evaluate_generalization.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType
from chakrview.runtime.tools import CalculatorTool

DATA_FILE = Path("experiments/batch351_380_adaptive_intelligence/data/randomized_generalization_benchmark.jsonl")
REPORTS_DIR = Path("experiments/batch351_380_adaptive_intelligence/reports")
CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
STEP150_CHECKPOINT = Path("experiments/batch311_350_neural_generalization/checkpoints/candidate_step150.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")


def run_evaluation():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(STEP150_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    router = CognitiveRouter()
    calc = CalculatorTool()

    items = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    neural_exact = 0
    neural_clean = 0
    runtime_exact = 0
    domain_breakdown = {}

    for item in items:
        dom = item["domain"]
        if dom not in domain_breakdown:
            domain_breakdown[dom] = {"total": 0, "neural_clean": 0, "runtime_handled": 0}
        domain_breakdown[dom]["total"] += 1

        # 1. Neural-Only generation
        prompt_tokens = [0] + tokenizer.encode(item["prompt"])
        gen = list(prompt_tokens)
        eos_reached = False
        with torch.no_grad():
            for _ in range(35):
                x = torch.tensor([gen[-256:]], dtype=torch.long)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt in (1, 2):
                    eos_reached = True
                    break
                gen.append(nxt)

        decoded = tokenizer.decode(gen[len(prompt_tokens):], errors="replace").strip()
        clean = decoded.replace("<|im_end|>", "").strip()
        expected = item["expected_answer"].strip()

        if "<|im_end|>" in decoded or eos_reached:
            neural_clean += 1
            domain_breakdown[dom]["neural_clean"] += 1

        if clean == expected:
            neural_exact += 1

        # 2. Integrated Runtime evaluation (Cognitive Router + Tools)
        decision = router.classify_route(item["query"])
        runtime_ans = None
        if decision.route == RouteType.CALCULATOR_TOOL:
            # Extract expression or clean query
            expr = item["query"]
            for prefix in ["What is ", "Calculate the value of ", "Compute "]:
                if expr.startswith(prefix):
                    expr = expr[len(prefix):]
            expr = expr.rstrip("?:. ")
            calc_res = calc.execute(expression=expr)
            if calc_res.success and str(calc_res.output).strip() == expected:
                runtime_exact += 1
                domain_breakdown[dom]["runtime_handled"] += 1
        elif decision.route == RouteType.DIRECT_CHAT:
            if clean == expected:
                runtime_exact += 1
                domain_breakdown[dom]["runtime_handled"] += 1

    summary = {
        "total_items": len(items),
        "neural_only": {
            "clean_termination_rate": round(neural_clean / len(items), 4),
            "exact_match_rate": round(neural_exact / len(items), 4),
        },
        "integrated_runtime": {
            "accuracy_rate": round(runtime_exact / len(items), 4),
        },
        "domain_breakdown": domain_breakdown,
    }

    out_file = REPORTS_DIR / "randomized_generalization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Neural Clean Term Rate: {summary['neural_only']['clean_termination_rate']*100:.1f}%")
    print(f"Integrated Runtime Accuracy: {summary['integrated_runtime']['accuracy_rate']*100:.1f}%")
    print(f"Results exported to: {out_file}")


if __name__ == "__main__":
    run_evaluation()
