"""
WP 970–974: End-to-End and Causal Evaluation on the Focused Task Family and Frozen Benchmark.
Compares:
1. Baseline Unified Fusion (Batch 901-940)
2. Improved Neural Topology Planner (Batch 941-980)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.span_grounding import ChakrSpanGroundingModel
from chakrview.cognition.neural_topology_planner import NeuralTopologyPlannerPipeline
from chakrview.cognition.plan_dsl import PlanDSLParser
from chakrview.cognition.unified_plan_fusion import NeuralSpanSlotFusionPipeline
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
SPAN_CKPT = REPO_ROOT / "experiments" / "batch861_900" / "checkpoints" / "candidate_step250_seed42_span_grounding.pt"
FOCUSED_TEST_PATH = REPO_ROOT / "experiments" / "batch941_980" / "data" / "frozen_test_focused_family.jsonl"
REPORT_DIR = REPO_ROOT / "experiments" / "batch941_980" / "reports"


def run_evaluations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512)
    span_model = ChakrSpanGroundingModel(cfg, max_slots=4)
    span_model.load_state_dict(torch.load(SPAN_CKPT, map_location=device, weights_only=False))
    span_model.to(device).eval()

    # Systems
    base_pipeline = NeuralSpanSlotFusionPipeline(span_model, tokenizer, device=device)
    improved_pipeline = NeuralTopologyPlannerPipeline(span_model, tokenizer, device=device)

    with open(FOCUSED_TEST_PATH, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f]

    total = len(test_items)
    base_exact = 0
    base_syntax = 0
    improved_exact = 0
    improved_syntax = 0

    print(f"Evaluating {total} held-out focused task family scenarios...")

    for it in test_items:
        prompt = it["prompt"]
        target = it["target_dsl"]

        # Baseline
        g_base, v_base, _ = base_pipeline.generate_plan(prompt)
        p_base = PlanDSLParser.parse(g_base)
        if v_base and p_base.is_valid: base_syntax += 1
        if g_base.strip() == target.strip(): base_exact += 1

        # Improved
        g_imp, v_imp, _ = improved_pipeline.generate_plan(prompt)
        p_imp = PlanDSLParser.parse(g_imp)
        if v_imp and p_imp.is_valid: improved_syntax += 1
        if g_imp.strip() == target.strip(): improved_exact += 1

    # Causal Tests (10 cases)
    causal_tests = [
        {"desc": "Change operand 600 -> 750", "prompt": "User: Calculate 750 * 5 and then add 14 to the result\nPlan:\n", "expect": "750*5"},
        {"desc": "Change operand 5 -> 8", "prompt": "User: Calculate 600 * 8 and then add 14 to the result\nPlan:\n", "expect": "600*8"},
        {"desc": "Swap operands 600*5 -> 5*600", "prompt": "User: Calculate 5 * 600 and then add 14 to the result\nPlan:\n", "expect": "5*600"},
        {"desc": "Repeated number 14", "prompt": "User: Calculate 14 * 5 and then add 14 to the result\nPlan:\n", "expect": "14*5"},
        {"desc": "Large number 1024", "prompt": "User: Calculate 1024 / 8 and then add 16 to the result\nPlan:\n", "expect": "1024/8"},
        {"desc": "Divide operation", "prompt": "User: Divide 800 by 4, then subtract 15\nPlan:\n", "expect": "800/4"},
        {"desc": "Hinglish phrasing", "prompt": "User: 600 aur 5 ko * karo aur result me 14 add karo\nPlan:\n", "expect": "600*5"},
        {"desc": "Missing account ID", "prompt": "User: Calculate net settlement without supplying account_id\nPlan:\n", "expect_clar": True},
        {"desc": "Missing auth signature", "prompt": "User: Execute transaction without specifying auth_signature\nPlan:\n", "expect_clar": True},
        {"desc": "Independent 3rd operand preservation", "prompt": "User: Calculate 600 * 5 and then add 99 to the result\nPlan:\n", "expect": "$0+99"},
    ]

    causal_results = []
    causal_passed = 0
    for ct in causal_tests:
        g, is_v, _ = improved_pipeline.generate_plan(ct["prompt"])
        if "expect_clar" in ct:
            passed = "CLARIFICATION: NEEDED" in g
        else:
            passed = ct["expect"] in g
        if passed: causal_passed += 1
        causal_results.append({
            "desc": ct["desc"],
            "passed": passed,
            "generated_plan": g.strip().replace("\n", " | "),
        })

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_items": total,
        "baseline_pipeline": {
            "syntax_validity": f"{base_syntax}/{total} ({round(base_syntax/total*100, 1)}%)",
            "exact_plan_match": f"{base_exact}/{total} ({round(base_exact/total*100, 1)}%)",
        },
        "improved_topology_pipeline": {
            "syntax_validity": f"{improved_syntax}/{total} ({round(improved_syntax/total*100, 1)}%)",
            "exact_plan_match": f"{improved_exact}/{total} ({round(improved_exact/total*100, 1)}%)",
        },
        "causal_evaluation": {
            "passed": f"{causal_passed}/{len(causal_tests)} ({round(causal_passed/len(causal_tests)*100, 1)}%)",
            "cases": causal_results,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "FOCUSED_EVALUATION_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("--- Focused Task Family Results ---")
    print("Baseline Exact Match:", f"{base_exact}/{total}")
    print("Improved Exact Match:", f"{improved_exact}/{total} ({round(improved_exact/total*100, 1)}%)")
    print(f"Causal Tests Passed: {causal_passed}/{len(causal_tests)}")


if __name__ == "__main__":
    run_evaluations()
