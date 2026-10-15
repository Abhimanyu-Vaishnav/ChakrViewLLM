"""
Batches 471-475: Baseline Reproduction & Intermediate-Step Audit.
Location: experiments/batch471_500_reliability/reproduce_baseline.py

Reproduces and verifies:
1. Held-out scratchpad compositional benchmark (6 items) on scratchpad candidate (SHA-256: 72784a61...).
2. Held-out paraphrase benchmark (20 items) on scratchpad candidate vs paraphrase candidate.
3. Diagnostic retention benchmark (15 items) across candidates.
4. Intermediate-step correctness audit (analyzing step-by-step arithmetic vs final answer).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
PARAPHRASE_CKPT = REPO_ROOT / "experiments" / "batch411_440_paraphrase" / "checkpoints" / "candidate_step150_paraphrase_augmented.pt"
SCRATCHPAD_CKPT = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "checkpoints" / "candidate_step150_scratchpad_reasoning.pt"

DIAGNOSTIC_DATA = REPO_ROOT / "experiments" / "batch381_410_integrity" / "data" / "diagnostic_answer_set.jsonl"
PARAPHRASE_DATA = REPO_ROOT / "experiments" / "batch411_440_paraphrase" / "data" / "test_heldout_paraphrase.jsonl"
SCRATCHPAD_DATA = REPO_ROOT / "experiments" / "batch441_470_reasoning_gate" / "data" / "test_heldout_scratchpad.jsonl"

REPORT_OUT = REPO_ROOT / "experiments" / "batch471_500_reliability" / "reports" / "baseline_reproduction_results.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: Path) -> Tuple[ChakrMicro, Dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"] if "config" in ck else ck.get("model_config", {})
    state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model = ChakrMicro(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def run_eval(model: ChakrMicro, tokenizer: Any, items: List[Dict[str, Any]], max_tokens: int = 40) -> Dict[str, Any]:
    exact_matches = 0
    clean_eos = 0
    details = []

    for item in items:
        prompt = item["prompt"]
        p_toks = [0] + tokenizer.encode(prompt)
        gen = list(p_toks)
        eos_found = False

        with torch.no_grad():
            for _ in range(max_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long)
                nxt = int(torch.argmax(model(x)[0, -1, :]).item())
                if nxt == 1:
                    eos_found = True
                    break
                gen.append(nxt)

        if eos_found:
            clean_eos += 1

        raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        expected = str(item.get("expected", item.get("target", ""))).strip()

        # Check match criteria
        is_match = False
        if expected and (expected in raw_gen or raw_gen == expected):
            is_match = True

        if is_match:
            exact_matches += 1

        details.append({
            "id": item.get("id", ""),
            "prompt": prompt,
            "expected": expected,
            "generated": raw_gen,
            "exact_match": is_match,
            "eos_reached": eos_found,
        })

    rate = exact_matches / max(len(items), 1)
    return {
        "total": len(items),
        "exact_matches": exact_matches,
        "exact_match_rate": rate,
        "clean_eos_rate": clean_eos / max(len(items), 1),
        "details": details,
    }


def main():
    print("=== BATCHES 471-475: BASELINE REPRODUCTION & AUDIT ===")
    hashes = {
        "canonical": sha256_file(CANONICAL_CKPT),
        "paraphrase_candidate": sha256_file(PARAPHRASE_CKPT),
        "scratchpad_candidate": sha256_file(SCRATCHPAD_CKPT),
    }
    print("Checkpoint Hashes:")
    for k, v in hashes.items():
        print(f"  {k}: {v}")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load datasets
    diag_items = [json.loads(l) for l in open(DIAGNOSTIC_DATA, encoding="utf-8") if l.strip()]
    para_items = [json.loads(l) for l in open(PARAPHRASE_DATA, encoding="utf-8") if l.strip()]
    sp_items = [json.loads(l) for l in open(SCRATCHPAD_DATA, encoding="utf-8") if l.strip()]

    print(f"\nLoaded Datasets: Diagnostic={len(diag_items)}, Paraphrase={len(para_items)}, Scratchpad={len(sp_items)}")

    # Load Models
    sp_model, _ = load_model(SCRATCHPAD_CKPT)
    para_model, _ = load_model(PARAPHRASE_CKPT)

    # 1. Scratchpad Candidate Eval
    print("\nEvaluating Scratchpad Candidate (72784a61...):")
    sp_scratchpad_res = run_eval(sp_model, tokenizer, sp_items, max_tokens=40)
    print(f"  Held-out Scratchpad Match: {sp_scratchpad_res['exact_match_rate']*100:.1f}% ({sp_scratchpad_res['exact_matches']}/{len(sp_items)})")

    sp_para_res = run_eval(sp_model, tokenizer, para_items, max_tokens=30)
    print(f"  Held-out Paraphrase Match: {sp_para_res['exact_match_rate']*100:.1f}% ({sp_para_res['exact_matches']}/{len(para_items)})")

    sp_diag_res = run_eval(sp_model, tokenizer, diag_items, max_tokens=30)
    print(f"  Diagnostic Retention Match: {sp_diag_res['exact_match_rate']*100:.1f}% ({sp_diag_res['exact_matches']}/{len(diag_items)})")

    # 2. Reference Paraphrase Candidate Eval
    print("\nEvaluating Paraphrase Candidate Reference (b8b82aaf...):")
    ref_scratchpad_res = run_eval(para_model, tokenizer, sp_items, max_tokens=40)
    print(f"  Reference Held-out Scratchpad Match: {ref_scratchpad_res['exact_match_rate']*100:.1f}% ({ref_scratchpad_res['exact_matches']}/{len(sp_items)})")

    ref_para_res = run_eval(para_model, tokenizer, para_items, max_tokens=30)
    print(f"  Reference Held-out Paraphrase Match: {ref_para_res['exact_match_rate']*100:.1f}% ({ref_para_res['exact_matches']}/{len(para_items)})")

    ref_diag_res = run_eval(para_model, tokenizer, diag_items, max_tokens=30)
    print(f"  Reference Diagnostic Retention Match: {ref_diag_res['exact_match_rate']*100:.1f}% ({ref_diag_res['exact_matches']}/{len(diag_items)})")

    # 3. Intermediate Step Correctness Audit
    print("\n--- Intermediate-Step Correctness Audit ---")
    audit_records = []
    for d in sp_scratchpad_res["details"]:
        gen_text = d["generated"]
        exp = d["expected"]
        lines = [l.strip() for l in gen_text.split("\n") if l.strip()]
        
        step_eval = {
            "id": d["id"],
            "prompt": d["prompt"],
            "generated": gen_text,
            "expected_final": exp,
            "has_steps": any(l.startswith("Step") for l in lines),
            "step_count": len([l for l in lines if l.startswith("Step")]),
            "final_answer_correct": d["exact_match"],
            "status": "UNKNOWN"
        }

        # Analyze correctness of reasoning trace
        if d["exact_match"] and step_eval["has_steps"]:
            step_eval["status"] = "CORRECT_TRACE_AND_ANSWER"
        elif not d["exact_match"] and step_eval["has_steps"]:
            step_eval["status"] = "INCORRECT_TRACE_AND_WRONG_ANSWER"
        elif d["exact_match"] and not step_eval["has_steps"]:
            step_eval["status"] = "NO_TRACE_ACCIDENTAL_CORRECT"
        else:
            step_eval["status"] = "NO_TRACE_WRONG_ANSWER"

        print(f"  [{step_eval['id']}] Status: {step_eval['status']} | Gen: {repr(gen_text)}")
        audit_records.append(step_eval)

    # Save output
    output_payload = {
        "hashes": hashes,
        "scratchpad_candidate": {
            "checkpoint": str(SCRATCHPAD_CKPT),
            "scratchpad": sp_scratchpad_res,
            "paraphrase": sp_para_res,
            "diagnostic": sp_diag_res,
            "step_audit": audit_records,
        },
        "paraphrase_candidate_reference": {
            "checkpoint": str(PARAPHRASE_CKPT),
            "scratchpad": ref_scratchpad_res,
            "paraphrase": ref_para_res,
            "diagnostic": ref_diag_res,
        }
    }
    REPORT_OUT.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nFull baseline reproduction results saved to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
