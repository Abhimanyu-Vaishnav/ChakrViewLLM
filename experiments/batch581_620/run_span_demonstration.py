"""
Batches 615-618: Controlled Capability Demonstration for Span Pointer Candidate.
Location: experiments/batch581_620/run_span_demonstration.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_intent_router import NeuralIntentRouter

CKPT_PATH = REPO_ROOT / "experiments" / "batch581_620" / "checkpoints" / "candidate_step250_span_pointer.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
DEMO_REPORT = REPO_ROOT / "experiments" / "batch581_620" / "reports" / "CAPABILITY_DEMONSTRATION.json"


def main():
    print("=== BATCHES 615-618: NEURAL SPAN EXTRACTION & GROUNDED DEMONSTRATION ===")
    router = NeuralIntentRouter(
        checkpoint_path=CKPT_PATH,
        tokenizer_dir=TOKENIZER_DIR,
    )

    test_queries = [
        "Please compute 36 / 12",
        "Can you compute 94 - 62?",
        "Run calculation for 52 + 67",
        "Convert 'indigenous architecture' to uppercase",
        "Multiply 75 by",
        "Run bash script: rm -rf /",
        "What is the capital city of planet Mars?",
        "Explain what an FPGA does in hardware",
        "Greetings assistant, good morning!",
    ]

    results = []

    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- [Turn {idx}] User: \"{q}\" ---")
        pred, raw_gen = router.predict_structured_intent(q)
        print(f"Neural Summary:       {raw_gen}")
        print(f"Predicted Intent:     {pred.get('intent')} | Tool: {pred.get('tool')} | Args: {pred.get('args')}")

        resp = router.process_turn(q)
        print(f"Final Output:         {resp.answer}")
        print(f"Component Origin:     {resp.component_origin.value}")
        print(f"Route Type:           {resp.route_decision.route.value}")
        print(f"Supported?:           {resp.is_supported}")

        results.append({
            "turn": idx,
            "query": q,
            "neural_summary": raw_gen,
            "parsed_intent": pred,
            "final_answer": resp.answer,
            "component_origin": resp.component_origin.value,
            "route_type": resp.route_decision.route.value,
            "is_supported": resp.is_supported,
            "support_explanation": resp.support_explanation,
            "latency_ms": round(resp.latency_ms, 2),
        })

    DEMO_REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved capability demonstration -> {DEMO_REPORT}")


if __name__ == "__main__":
    main()
