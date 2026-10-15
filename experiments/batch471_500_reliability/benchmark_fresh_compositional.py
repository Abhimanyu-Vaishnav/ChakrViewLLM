"""
Batches 476-482: Fresh Compositional Benchmark Generator and Evaluation.
Location: experiments/batch471_500_reliability/benchmark_fresh_compositional.py

Constructs:
1. Fresh held-out task families:
   - Compound arithmetic with novel digits.
   - Distinct operator orderings (e.g. A - B + C, (A - B) * C).
   - Word problems with intermediate reasoning.
   - Direct single-step controls.
   - Paraphrased wording variants.
2. Evaluates scratchpad candidate (neural-only, zero tools) against ground-truth calculator.
3. Categorizes responses:
   - CORRECT_TRACE_AND_ANSWER
   - CORRECT_TRACE_WRONG_ANSWER
   - WRONG_TRACE_ACCIDENTAL_CORRECT
   - WRONG_TRACE_AND_ANSWER
   - REPLAY_SHORTCUT_DETECTED
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
SCRATCHPAD_CKPT = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "checkpoints" / "candidate_step150_scratchpad_reasoning.pt"

DATA_DIR = REPO_ROOT / "experiments" / "batch471_500_reliability" / "data"
REPORT_DIR = REPO_ROOT / "experiments" / "batch471_500_reliability" / "reports"

# 12 Fresh Compositional Tasks
FRESH_TASKS = [
    # Family 1: Novel addition & subtraction operands
    {
        "id": "fresh_comp_001",
        "family": "add_sub_novel",
        "prompt": "Question: Calculate (16 plus 14) minus 12.\nReasoning:\n",
        "expected_final": "18",
        "expected_steps": ["16 + 14 = 30", "30 - 12 = 18"],
    },
    {
        "id": "fresh_comp_002",
        "family": "add_sub_novel",
        "prompt": "Question: What is (40 minus 15) plus 5?\nReasoning:\n",
        "expected_final": "30",
        "expected_steps": ["40 - 15 = 25", "25 + 5 = 30"],
    },
    # Family 2: Novel multiplication & subtraction operands
    {
        "id": "fresh_comp_003",
        "family": "mul_sub_novel",
        "prompt": "Question: What is 5 multiplied by 8 minus 15?\nReasoning:\n",
        "expected_final": "25",
        "expected_steps": ["5 * 8 = 40", "40 - 15 = 25"],
    },
    {
        "id": "fresh_comp_004",
        "family": "mul_sub_novel",
        "prompt": "Question: Calculate 9 multiplied by 4 minus 6.\nReasoning:\n",
        "expected_final": "30",
        "expected_steps": ["9 * 4 = 36", "36 - 6 = 30"],
    },
    # Family 3: Novel square function composition
    {
        "id": "fresh_comp_005",
        "family": "add_sq_novel",
        "prompt": "Question: Calculate the square of (1 plus 6).\nReasoning:\n",
        "expected_final": "49",
        "expected_steps": ["1 + 6 = 7", "7 * 7 = 49"],
    },
    {
        "id": "fresh_comp_006",
        "family": "add_sq_novel",
        "prompt": "Question: What is the square of (4 plus 5)?\nReasoning:\n",
        "expected_final": "81",
        "expected_steps": ["4 + 5 = 9", "9 * 9 = 81"],
    },
    # Family 4: Reordered operator composition (Subtraction before addition)
    {
        "id": "fresh_comp_007",
        "family": "reordered_sub_add",
        "prompt": "Question: Calculate 50 minus 20 plus 8.\nReasoning:\n",
        "expected_final": "38",
        "expected_steps": ["50 - 20 = 30", "30 + 8 = 38"],
    },
    # Family 5: Novel Geometry angles
    {
        "id": "fresh_comp_008",
        "family": "geometry_angle_novel",
        "prompt": "Question: In a triangle, if two angles are 50 and 80 degrees, what is the third angle?\nReasoning:\n",
        "expected_final": "50",
        "expected_steps": ["50 + 80 = 130", "180 - 130 = 50"],
    },
    # Family 6: Word problem to arithmetic mapping
    {
        "id": "fresh_comp_009",
        "family": "word_problem_comp",
        "prompt": "Question: A baker has 12 apples, buys 8 more, and uses 5. How many are left?\nReasoning:\n",
        "expected_final": "15",
        "expected_steps": ["12 + 8 = 20", "20 - 5 = 15"],
    },
    # Family 7: Direct single-step controls (positive baseline control)
    {
        "id": "fresh_comp_010",
        "family": "single_step_control",
        "prompt": "Question: What is 14 plus 28?\nAnswer:",
        "expected_final": "42",
        "expected_steps": [],
    },
    {
        "id": "fresh_comp_011",
        "family": "single_step_control",
        "prompt": "Question: Calculate 7 multiplied by 6.\nAnswer:",
        "expected_final": "42",
        "expected_steps": [],
    },
    # Family 8: Hindi paraphrased arithmetic task
    {
        "id": "fresh_comp_012",
        "family": "hindi_arithmetic_variant",
        "prompt": "Question: 14 aur 28 ka jod kya hai?\nAnswer:",
        "expected_final": "42",
        "expected_steps": [],
    },
]


def run_benchmark():
    # 1. Save benchmark manifest
    manifest_file = DATA_DIR / "fresh_compositional_benchmark.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for t in FRESH_TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Fresh benchmark manifest written to: {manifest_file}")

    # 2. Load model & tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(SCRATCHPAD_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    results = []
    correct_final_count = 0
    correct_trace_count = 0
    replay_count = 0

    print("\n--- Running Evaluation on Fresh Compositional Benchmark ---")
    for task in FRESH_TASKS:
        prompt = task["prompt"]
        p_toks = [0] + tokenizer.encode(prompt)
        gen = list(p_toks)
        eos_found = False

        with torch.no_grad():
            for _ in range(45):
                x = torch.tensor([gen[-256:]], dtype=torch.long)
                nxt = int(torch.argmax(model(x)[0, -1, :]).item())
                if nxt == 1:
                    eos_found = True
                    break
                gen.append(nxt)

        raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        exp_final = task["expected_final"]

        # Check final answer match
        final_match = (exp_final in raw_gen) or (raw_gen == exp_final)
        if final_match:
            correct_final_count += 1

        # Check if model produced a memorized training trace shortcut (e.g. "14 + 28 = 42" or "7 * 6 = 42")
        is_replay = False
        if "14 + 28 = 42" in raw_gen and "14" not in prompt and "28" not in prompt:
            is_replay = True
            replay_count += 1
        elif "7 * 6 = 42" in raw_gen and "7" not in prompt and "6" not in prompt:
            is_replay = True
            replay_count += 1

        # Check trace correctness
        trace_valid = False
        if task["expected_steps"]:
            # Check if all intermediate steps appear
            trace_valid = all(step.replace(" ", "") in raw_gen.replace(" ", "") for step in task["expected_steps"])
            if trace_valid:
                correct_trace_count += 1

        # Classify status
        if is_replay:
            status = "REPLAY_SHORTCUT_DETECTED"
        elif final_match and trace_valid:
            status = "CORRECT_TRACE_AND_ANSWER"
        elif final_match and not task["expected_steps"]:
            status = "DIRECT_CORRECT_ANSWER"
        elif final_match and not trace_valid:
            status = "WRONG_TRACE_ACCIDENTAL_CORRECT"
        elif not final_match and trace_valid:
            status = "CORRECT_TRACE_WRONG_ANSWER"
        else:
            status = "WRONG_TRACE_AND_ANSWER"

        print(f"[{task['id']}] {task['family']} | exp={exp_final} status={status} eos={eos_found}\n  Generated: {repr(raw_gen)}")

        results.append({
            "id": task["id"],
            "family": task["family"],
            "prompt": prompt,
            "expected_final": exp_final,
            "expected_steps": task["expected_steps"],
            "generated": raw_gen,
            "final_match": final_match,
            "trace_valid": trace_valid,
            "is_replay_shortcut": is_replay,
            "status": status,
            "eos_reached": eos_found,
        })

    total = len(FRESH_TASKS)
    final_rate = correct_final_count / total
    out_report = {
        "benchmark_size": total,
        "correct_final_answers": correct_final_count,
        "final_accuracy": final_rate,
        "correct_traces": correct_trace_count,
        "replay_shortcuts_detected": replay_count,
        "results": results,
    }

    report_path = REPORT_DIR / "fresh_compositional_results.json"
    report_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(f"\nFresh Compositional Results written to: {report_path}")
    print(f"Final Accuracy: {final_rate*100:.1f}% ({correct_final_count}/{total})")
    print(f"Replay Shortcut Rate: {replay_count}/{total} ({replay_count/total*100:.1f}%)")


if __name__ == "__main__":
    run_benchmark()
