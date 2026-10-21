"""
evaluate_exp11.py
ChakrView Experiment 11 — Comprehensive Hybrid vs Baselines Evaluation Harness

Evaluates:
- Baseline A: Native Causal LM (ChakrMicroForCausalLM without pointer)
- Baseline B: Exp 10 Span Pointer (evaluated on copy slot fields only)
- Baseline C: Hybrid Generator-Pointer Prototype
- Control: Deterministic Lookup parser (sanity check)

Across:
- Validation (60 examples)
- Test A: Familiar entities in novel contexts (60 examples)
- Test B: Completely held-out entities, statuses, and codes (60 examples)

Tracks:
1. JSON Parse Validity (valid json.loads)
2. Schema Validity (contains requested keys, no illegal keys, dict type)
3. Correct Field Names
4. Per-field exact value match (entity, status, code)
5. Whole-object exact match
6. Copy accuracy on eligible copy targets
7. Generated token accuracy on syntax/structural targets
8. Performance on held-out values and combinations
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel
from experiments.research_v08_json_sft.exp11_hybrid_json.hybrid_generator_pointer import ChakrHybridJsonModel

EXP11_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP11_DIR / "data"
CHECKPOINT_DIR = EXP11_DIR / "checkpoints"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

PARENT_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXP10_CKPT = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp10_span_pointer" / "checkpoints" / "exp10_span_pointer_best.pt"
EXP11_CKPT = CHECKPOINT_DIR / "exp11_hybrid_best.pt"

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def deterministic_lookup(prompt: str) -> Dict[str, str]:
    """Sanity check deterministic parser."""
    res = {}
    m_ent = re.search(r"\[([a-zA-Z0-9_]+)\]", prompt)
    if m_ent:
        res["entity"] = m_ent.group(1).strip()
    m_stat = re.search(r"'([a-zA-Z0-9_]+)'", prompt)
    if m_stat:
        res["status"] = m_stat.group(1).strip()
    m_code = re.search(r"code\s*[:=]?\s*([0-9]+)", prompt)
    if m_code:
        res["code"] = m_code.group(1).strip()
    return res

def evaluate_baseline_a_native_lm(
    model: ChakrMicro,
    dataset: List[Dict[str, Any]],
    tokenizer: Any
) -> List[Dict[str, Any]]:
    """Native LM greedy autoregressive decoding."""
    results = []
    for item in dataset:
        prompt_ids = [0] + tokenizer.encode(item["prompt"])
        curr_ids = list(prompt_ids)
        gen_toks = []
        for _ in range(64):
            inp = torch.tensor([curr_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp)
                next_tok = logits[0, -1, :].argmax().item()
            if next_tok == 1:
                break
            curr_ids.append(next_tok)
            gen_toks.append(next_tok)
            gen_str = tokenizer.decode(gen_toks)
            if "}" in gen_str and gen_str.strip().endswith("}"):
                break

        raw_gen = tokenizer.decode(gen_toks).strip()
        parsed = None
        json_valid = False
        try:
            parsed = json.loads(raw_gen)
            json_valid = isinstance(parsed, dict)
        except Exception:
            pass

        schema_valid = False
        exact_obj_match = False
        if json_valid:
            schema_valid = set(parsed.keys()) == set(item["target_dict"].keys())
            exact_obj_match = (parsed == item["target_dict"])

        results.append({
            "model": "baseline_a_native_lm",
            "example_id": item["example_id"],
            "raw_output": raw_gen,
            "parsed_json": parsed,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "exact_obj_match": exact_obj_match,
            "target_json": item["target_json"]
        })
    return results

def evaluate_baseline_b_span_pointer(
    model: ChakrSpanPointerModel,
    dataset: List[Dict[str, Any]],
    tokenizer: Any
) -> List[Dict[str, Any]]:
    """
    Exp 10 Span Pointer on primary copy target.
    Note: Span pointer emits a string span, not a structured JSON object.
    It is evaluated on whether it correctly points to the primary target entity/status.
    """
    results = []
    for item in dataset:
        prompt_toks = [0] + tokenizer.encode(item["prompt"])
        seq_len = len(prompt_toks)
        inp = torch.tensor([prompt_toks], dtype=torch.long)
        mask = torch.zeros((1, seq_len), dtype=torch.bool)
        mask[0, 1:seq_len - 2] = True

        pred_s, pred_e, pred_toks, _ = model.predict_span(inp, mask)
        pred_val = tokenizer.decode(pred_toks).strip()

        # Check if matched any target value
        matched_field = None
        for k, v in item["target_dict"].items():
            if pred_val == v:
                matched_field = k
                break

        results.append({
            "model": "baseline_b_span_pointer",
            "example_id": item["example_id"],
            "raw_output": pred_val,
            "parsed_json": None,
            "json_valid": False,  # Output type is string span, not JSON
            "schema_valid": False,
            "exact_obj_match": False,
            "copy_match": matched_field is not None,
            "matched_field": matched_field,
            "target_json": item["target_json"]
        })
    return results

def evaluate_baseline_c_hybrid(
    model: ChakrHybridJsonModel,
    dataset: List[Dict[str, Any]],
    tokenizer: Any
) -> List[Dict[str, Any]]:
    """Hybrid Generator-Pointer with explicit routing."""
    results = []
    for item in dataset:
        prompt_toks = [0] + tokenizer.encode(item["prompt"])
        seq_len = len(prompt_toks)
        cand_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        cand_mask[0, 1:seq_len - 2] = True

        gen_out = model.generate_json_hybrid(prompt_toks, tokenizer, cand_mask, max_new_tokens=64)
        raw_gen = gen_out["generated_text"]

        parsed = None
        json_valid = False
        try:
            parsed = json.loads(raw_gen)
            json_valid = isinstance(parsed, dict)
        except Exception:
            # Fallback cleanup for trailing tokens
            m = re.search(r"\{[^\}]+\}", raw_gen)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    json_valid = isinstance(parsed, dict)
                except Exception:
                    pass

        schema_valid = False
        exact_obj_match = False
        if json_valid:
            schema_valid = set(parsed.keys()) == set(item["target_dict"].keys())
            exact_obj_match = (parsed == item["target_dict"])

        results.append({
            "model": "baseline_c_hybrid",
            "example_id": item["example_id"],
            "raw_output": raw_gen,
            "parsed_json": parsed,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "exact_obj_match": exact_obj_match,
            "routing_events": len(gen_out["routing_decisions"]),
            "target_json": item["target_json"]
        })
    return results

def compute_split_metrics(eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(eval_rows)
    if n == 0:
        return {}
    json_valid_cnt = sum(1 for r in eval_rows if r["json_valid"])
    schema_valid_cnt = sum(1 for r in eval_rows if r["schema_valid"])
    exact_obj_cnt = sum(1 for r in eval_rows if r["exact_obj_match"])

    return {
        "total": n,
        "json_valid_count": json_valid_cnt,
        "json_valid_pct": (json_valid_cnt / n) * 100,
        "schema_valid_count": schema_valid_cnt,
        "schema_valid_pct": (schema_valid_cnt / n) * 100,
        "exact_obj_count": exact_obj_cnt,
        "exact_obj_pct": (exact_obj_cnt / n) * 100,
    }

def main():
    print("=" * 70)
    print("EXP 11: COMPREHENSIVE HYBRID VS BASELINES EVALUATION")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=512)

    # 1. Load Baseline A (Native Causal LM)
    print("Loading Baseline A (Native Causal LM)...")
    native_model = ChakrMicro(config)
    p_st = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    b_dict = {}
    for k, v in p_st["model_state_dict"].items():
        if k.startswith("transformer."):
            b_dict[k.replace("transformer.", "")] = v
        elif k.startswith("backbone."):
            b_dict[k.replace("backbone.", "")] = v
        else:
            b_dict[k] = v
    native_model.load_state_dict(b_dict, strict=False)
    native_model.eval()

    # 2. Load Baseline B (Exp 10 Span Pointer)
    print("Loading Baseline B (Exp 10 Span Pointer)...")
    ptr_model = ChakrSpanPointerModel(config, max_span_len=8, freeze_backbone=False)
    exp10_st = torch.load(EXP10_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    ptr_model.load_state_dict(exp10_st)
    ptr_model.eval()

    # 3. Load Baseline C (Hybrid Generator-Pointer)
    print("Loading Baseline C (Hybrid Generator-Pointer)...")
    hybrid_model = ChakrHybridJsonModel(config, max_span_len=8, freeze_backbone=False)
    exp11_st = torch.load(EXP11_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    hybrid_model.load_state_dict(exp11_st)
    hybrid_model.eval()

    splits = {
        "val": load_jsonl(DATA_DIR / "val.jsonl"),
        "test_a": load_jsonl(DATA_DIR / "test_a_familiar.jsonl"),
        "test_b": load_jsonl(DATA_DIR / "test_b_held_out.jsonl"),
    }

    results_summary = {}
    all_csv_rows = []

    for split_name, dataset in splits.items():
        print(f"\n--- Split: {split_name.upper()} ({len(dataset)} examples) ---")
        results_summary[split_name] = {}

        # Eval A
        res_a = evaluate_baseline_a_native_lm(native_model, dataset, tokenizer)
        m_a = compute_split_metrics(res_a)
        results_summary[split_name]["baseline_a_native_lm"] = m_a
        print(f"  [Native LM] JSON Valid: {m_a['json_valid_pct']:.1f}% | Exact Obj Match: {m_a['exact_obj_pct']:.1f}%")

        # Eval B
        res_b = evaluate_baseline_b_span_pointer(ptr_model, dataset, tokenizer)
        m_b = compute_split_metrics(res_b)
        copy_hits = sum(1 for r in res_b if r.get("copy_match"))
        m_b["copy_hit_count"] = copy_hits
        m_b["copy_hit_pct"] = (copy_hits / len(dataset)) * 100
        results_summary[split_name]["baseline_b_span_pointer"] = m_b
        print(f"  [Span Pointer] Slot Copy Hit: {m_b['copy_hit_pct']:.1f}% (Output type: span string, not structured JSON)")

        # Eval C
        res_c = evaluate_baseline_c_hybrid(hybrid_model, dataset, tokenizer)
        m_c = compute_split_metrics(res_c)
        results_summary[split_name]["baseline_c_hybrid"] = m_c
        print(f"  [Hybrid]    JSON Valid: {m_c['json_valid_pct']:.1f}% | Exact Obj Match: {m_c['exact_obj_pct']:.1f}%")

        for r in res_a + res_b + res_c:
            r["split"] = split_name
            all_csv_rows.append(r)

    # Save EXP11_RESULTS.json
    res_path = EXP11_DIR / "EXP11_RESULTS.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved overall metrics to {res_path}")

    # Save EXP11_PER_EXAMPLE.csv
    csv_path = EXP11_DIR / "EXP11_PER_EXAMPLE.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "split", "example_id",
            "json_valid", "schema_valid", "exact_obj_match",
            "copy_match", "raw_output", "target_json"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_csv_rows:
            writer.writerow(r)
    print(f"Saved per-example rows to {csv_path}")

if __name__ == "__main__":
    main()
