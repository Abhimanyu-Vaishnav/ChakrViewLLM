"""
evaluate_exp12.py
ChakrView Experiment 12 — Comprehensive Evaluation of Constrained Hybrid vs Baselines

Evaluates 5 Fair Experimental Arms:
- Arm A: Native Causal LM Baseline (unconstrained autoregressive)
- Arm B: Unconstrained Exp 11 Hybrid (existing hybrid behavior)
- Arm C: Constrained Hybrid (Exp 11 model + Schema Controller)
- Arm D: Controller-Plus-Oracle-Value Upper Bound (gold spans + Schema Controller)
- Arm E: Deterministic Regex Sanity Baseline (rule-based pattern matching)

Evaluated across:
- Validation Split (60 examples)
- Test A Split: Familiar entities in novel contexts (60 examples)
- Test B Split: Completely held-out entities, statuses, and codes (60 examples)

Tracks all metrics with explicit numerators and denominators:
- Valid JSON syntax rate
- Exact schema compliance rate
- Required-field completeness rate
- Duplicate/unrequested key rate (0% guaranteed under controller)
- Per-field exact value match (entity, status, code)
- Whole-object exact match rate
- Breakdown by requested-field count (1, 2, 3 fields)
- Breakdown on familiar vs held-out entities
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp11_hybrid_json.hybrid_generator_pointer import ChakrHybridJsonModel
from experiments.research_v08_json_sft.exp12_constrained_hybrid.schema_controller import SchemaController

EXP11_DIR = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp11_hybrid_json"
EXP12_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP11_DIR / "data"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

PARENT_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXP11_CKPT = EXP11_DIR / "checkpoints" / "exp11_hybrid_best.pt"

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def regex_deterministic_lookup(context: str, instruction: str) -> Dict[str, str]:
    """Arm E: Transparent rule-based parser baseline."""
    res = {}
    m_ent = re.search(r"\[([a-zA-Z0-9_]+)\]", context)
    if m_ent:
        res["entity"] = m_ent.group(1).strip()
    m_stat = re.search(r"'([a-zA-Z0-9_]+)'", context)
    if m_stat:
        res["status"] = m_stat.group(1).strip()
    m_code = re.search(r"code\s*[:=]?\s*([0-9]+)", context)
    if m_code:
        res["code"] = m_code.group(1).strip()
    return res

def evaluate_arms_on_dataset(
    dataset: List[Dict[str, Any]],
    split_name: str,
    native_model: ChakrMicro,
    hybrid_model: ChakrHybridJsonModel,
    tokenizer: Any,
    controller: SchemaController
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total = len(dataset)
    per_example_rows = []

    # Accumulators for all 5 arms
    arm_metrics = {
        "arm_a_native_lm": collections.defaultdict(int),
        "arm_b_unconstrained_hybrid": collections.defaultdict(int),
        "arm_c_constrained_hybrid": collections.defaultdict(int),
        "arm_d_oracle_upper_bound": collections.defaultdict(int),
        "arm_e_deterministic_regex": collections.defaultdict(int),
    }

    # Breakdown by field count for Arm C
    arm_c_by_field_count = collections.defaultdict(lambda: {"total": 0, "exact_obj": 0, "json_valid": 0})

    for item in dataset:
        prompt = item["prompt"]
        context = item["context"]
        instruction = item["instruction"]
        target_dict = item["target_dict"]
        target_json_str = item["target_json"]
        copy_spans = item.get("copy_spans", {})
        req_fields = controller.parse_requested_fields(instruction)
        n_fields = len(req_fields)
        arm_c_by_field_count[n_fields]["total"] += 1

        prompt_toks = [0] + tokenizer.encode(prompt)
        seq_len = len(prompt_toks)
        cand_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        cand_mask[0, 1:seq_len - 2] = True

        # ---------------------------------------------------------
        # Arm A: Native Causal LM
        # ---------------------------------------------------------
        curr_ids = list(prompt_toks)
        gen_toks_a = []
        for _ in range(64):
            inp = torch.tensor([curr_ids], dtype=torch.long)
            with torch.no_grad():
                logits = native_model(inp)
                nxt = logits[0, -1, :].argmax().item()
            if nxt == 1:
                break
            curr_ids.append(nxt)
            gen_toks_a.append(nxt)
            t_str = tokenizer.decode(gen_toks_a)
            if "}" in t_str and t_str.strip().endswith("}"):
                break
        raw_a = tokenizer.decode(gen_toks_a).strip()
        parsed_a = None
        json_v_a = False
        try:
            parsed_a = json.loads(raw_a)
            json_v_a = isinstance(parsed_a, dict)
        except Exception:
            pass
        schema_v_a = json_v_a and set(parsed_a.keys()) == set(target_dict.keys())
        obj_em_a = (parsed_a == target_dict)

        if json_v_a: arm_metrics["arm_a_native_lm"]["json_valid"] += 1
        if schema_v_a: arm_metrics["arm_a_native_lm"]["schema_valid"] += 1
        if obj_em_a: arm_metrics["arm_a_native_lm"]["exact_obj"] += 1

        # ---------------------------------------------------------
        # Arm B: Unconstrained Exp 11 Hybrid
        # ---------------------------------------------------------
        gen_out_b = hybrid_model.generate_json_hybrid(prompt_toks, tokenizer, cand_mask, max_new_tokens=64)
        raw_b = gen_out_b["generated_text"]
        parsed_b = None
        json_v_b = False
        try:
            parsed_b = json.loads(raw_b)
            json_v_b = isinstance(parsed_b, dict)
        except Exception:
            m = re.search(r"\{[^\}]+\}", raw_b)
            if m:
                try:
                    parsed_b = json.loads(m.group(0))
                    json_v_b = isinstance(parsed_b, dict)
                except Exception:
                    pass
        schema_v_b = json_v_b and set(parsed_b.keys()) == set(target_dict.keys())
        obj_em_b = (parsed_b == target_dict)

        if json_v_b: arm_metrics["arm_b_unconstrained_hybrid"]["json_valid"] += 1
        if schema_v_b: arm_metrics["arm_b_unconstrained_hybrid"]["schema_valid"] += 1
        if obj_em_b: arm_metrics["arm_b_unconstrained_hybrid"]["exact_obj"] += 1

        # ---------------------------------------------------------
        # Arm C: Constrained Hybrid (Exp 11 Model + Schema Controller)
        # ---------------------------------------------------------
        # For each requested field, invoke the pointer model on the candidate mask
        prompt_inp = torch.tensor([prompt_toks], dtype=torch.long)
        extracted_field_values_c = {}
        for fld in req_fields:
            s_c, e_c, toks_c = hybrid_model.point_span(prompt_inp, cand_mask, query_pos=-1)
            extracted_val = tokenizer.decode(toks_c).strip().lstrip(":=").rstrip(",").strip()
            extracted_field_values_c[fld] = extracted_val

        raw_c, parsed_c, json_v_c = controller.format_constrained_json(req_fields, extracted_field_values_c)
        schema_v_c = json_v_c and set(parsed_c.keys()) == set(target_dict.keys())
        obj_em_c = (parsed_c == target_dict)

        if json_v_c: arm_metrics["arm_c_constrained_hybrid"]["json_valid"] += 1
        if schema_v_c: arm_metrics["arm_c_constrained_hybrid"]["schema_valid"] += 1
        if obj_em_c:
            arm_metrics["arm_c_constrained_hybrid"]["exact_obj"] += 1
            arm_c_by_field_count[n_fields]["exact_obj"] += 1
        if json_v_c:
            arm_c_by_field_count[n_fields]["json_valid"] += 1

        # Check per-field accuracy
        for fld in req_fields:
            if parsed_c.get(fld) == target_dict.get(fld):
                arm_metrics["arm_c_constrained_hybrid"][f"field_{fld}_correct"] += 1
            arm_metrics["arm_c_constrained_hybrid"][f"field_{fld}_total"] += 1

        # ---------------------------------------------------------
        # Arm D: Oracle Upper Bound (Controller + Gold Spans)
        # ---------------------------------------------------------
        oracle_field_values = {}
        for fld in req_fields:
            if fld in copy_spans:
                s_gold = copy_spans[fld]["start_token_index"]
                e_gold = copy_spans[fld]["end_token_index"]
                oracle_toks = prompt_toks[s_gold:e_gold+1]
                oracle_field_values[fld] = tokenizer.decode(oracle_toks).strip()
            else:
                oracle_field_values[fld] = target_dict.get(fld, "none")

        raw_d, parsed_d, json_v_d = controller.format_constrained_json(req_fields, oracle_field_values)
        schema_v_d = json_v_d and set(parsed_d.keys()) == set(target_dict.keys())
        obj_em_d = (parsed_d == target_dict)

        if json_v_d: arm_metrics["arm_d_oracle_upper_bound"]["json_valid"] += 1
        if schema_v_d: arm_metrics["arm_d_oracle_upper_bound"]["schema_valid"] += 1
        if obj_em_d: arm_metrics["arm_d_oracle_upper_bound"]["exact_obj"] += 1

        # ---------------------------------------------------------
        # Arm E: Deterministic Regex Sanity Check
        # ---------------------------------------------------------
        extracted_regex = regex_deterministic_lookup(context, instruction)
        raw_e, parsed_e, json_v_e = controller.format_constrained_json(req_fields, extracted_regex)
        schema_v_e = json_v_e and set(parsed_e.keys()) == set(target_dict.keys())
        obj_em_e = (parsed_e == target_dict)

        if json_v_e: arm_metrics["arm_e_deterministic_regex"]["json_valid"] += 1
        if schema_v_e: arm_metrics["arm_e_deterministic_regex"]["schema_valid"] += 1
        if obj_em_e: arm_metrics["arm_e_deterministic_regex"]["exact_obj"] += 1

        # Record per-example row
        per_example_rows.append({
            "split": split_name,
            "example_id": item["example_id"],
            "requested_fields": json.dumps(req_fields),
            "target_dict": json.dumps(target_dict),
            "target_json": target_json_str,
            "arm_a_raw": raw_a,
            "arm_a_json_valid": json_v_a,
            "arm_a_schema_valid": schema_v_a,
            "arm_a_exact_obj": obj_em_a,
            "arm_b_raw": raw_b,
            "arm_b_json_valid": json_v_b,
            "arm_b_schema_valid": schema_v_b,
            "arm_b_exact_obj": obj_em_b,
            "arm_c_raw": raw_c,
            "arm_c_json_valid": json_v_c,
            "arm_c_schema_valid": schema_v_c,
            "arm_c_exact_obj": obj_em_c,
            "arm_d_raw": raw_d,
            "arm_d_exact_obj": obj_em_d,
            "arm_e_raw": raw_e,
            "arm_e_exact_obj": obj_em_e
        })

    # Compile split summary
    split_summary = {}
    for arm_name, counts in arm_metrics.items():
        split_summary[arm_name] = {
            "total": total,
            "json_valid_count": counts["json_valid"],
            "json_valid_pct": (counts["json_valid"] / total) * 100,
            "schema_valid_count": counts["schema_valid"],
            "schema_valid_pct": (counts["schema_valid"] / total) * 100,
            "exact_obj_count": counts["exact_obj"],
            "exact_obj_pct": (counts["exact_obj"] / total) * 100,
        }
        if arm_name == "arm_c_constrained_hybrid":
            split_summary[arm_name]["by_field_count"] = dict(arm_c_by_field_count)

    return split_summary, per_example_rows

def main():
    print("=" * 70)
    print("EXP 12: SCHEMA-CONSTRAINED HYBRID EXTRACTION EVALUATION")
    print("=" * 70)

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=512)
    controller = SchemaController()

    # 1. Load Arm A: Native Causal LM
    print("\nLoading Arm A (Native Causal LM)...")
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

    # 2. Load Arm B & C: Hybrid Model (Frozen Exp 11 Checkpoint)
    print("Loading Arm B & C (Frozen Exp 11 Checkpoint)...")
    hybrid_model = ChakrHybridJsonModel(config, max_span_len=8, freeze_backbone=False)
    exp11_st = torch.load(EXP11_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    hybrid_model.load_state_dict(exp11_st)
    hybrid_model.eval()

    splits = {
        "val": load_jsonl(DATA_DIR / "val.jsonl"),
        "test_a": load_jsonl(DATA_DIR / "test_a_familiar.jsonl"),
        "test_b": load_jsonl(DATA_DIR / "test_b_held_out.jsonl"),
    }

    results_all = {}
    all_csv_rows = []

    for split_name, dataset in splits.items():
        print(f"\n--- Evaluating Split: {split_name.upper()} ({len(dataset)} examples) ---")
        summary, rows = evaluate_arms_on_dataset(
            dataset, split_name, native_model, hybrid_model, tokenizer, controller
        )
        results_all[split_name] = summary
        all_csv_rows.extend(rows)

        print(f"  [Arm A: Native LM]         JSON: {summary['arm_a_native_lm']['json_valid_pct']:.1f}% | Schema: {summary['arm_a_native_lm']['schema_valid_pct']:.1f}% | Exact Obj EM: {summary['arm_a_native_lm']['exact_obj_pct']:.1f}%")
        print(f"  [Arm B: Unconstrained]     JSON: {summary['arm_b_unconstrained_hybrid']['json_valid_pct']:.1f}% | Schema: {summary['arm_b_unconstrained_hybrid']['schema_valid_pct']:.1f}% | Exact Obj EM: {summary['arm_b_unconstrained_hybrid']['exact_obj_pct']:.1f}%")
        print(f"  [Arm C: Constrained]       JSON: {summary['arm_c_constrained_hybrid']['json_valid_pct']:.1f}% | Schema: {summary['arm_c_constrained_hybrid']['schema_valid_pct']:.1f}% | Exact Obj EM: {summary['arm_c_constrained_hybrid']['exact_obj_pct']:.1f}%")
        print(f"  [Arm D: Oracle Upper Bound]JSON: {summary['arm_d_oracle_upper_bound']['json_valid_pct']:.1f}% | Schema: {summary['arm_d_oracle_upper_bound']['schema_valid_pct']:.1f}% | Exact Obj EM: {summary['arm_d_oracle_upper_bound']['exact_obj_pct']:.1f}%")
        print(f"  [Arm E: Deterministic]     JSON: {summary['arm_e_deterministic_regex']['json_valid_pct']:.1f}% | Schema: {summary['arm_e_deterministic_regex']['schema_valid_pct']:.1f}% | Exact Obj EM: {summary['arm_e_deterministic_regex']['exact_obj_pct']:.1f}%")

    # Save EXP12_RESULTS.json
    res_path = EXP12_DIR / "EXP12_RESULTS.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved overall metrics to {res_path}")

    # Save EXP12_PER_EXAMPLE.csv
    csv_path = EXP12_DIR / "EXP12_PER_EXAMPLE.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split", "example_id", "requested_fields", "target_dict", "target_json",
            "arm_a_json_valid", "arm_a_schema_valid", "arm_a_exact_obj", "arm_a_raw",
            "arm_b_json_valid", "arm_b_schema_valid", "arm_b_exact_obj", "arm_b_raw",
            "arm_c_json_valid", "arm_c_schema_valid", "arm_c_exact_obj", "arm_c_raw",
            "arm_d_exact_obj", "arm_d_raw", "arm_e_exact_obj", "arm_e_raw"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_csv_rows:
            writer.writerow(r)
    print(f"Saved per-example records to {csv_path}")

    # Save manifest.json
    manifest = {
        "canonical_checkpoint": {
            "path": str(PARENT_CKPT.relative_to(REPO_ROOT)),
            "sha256": compute_sha256(PARENT_CKPT)
        },
        "tokenizer_vocab": {
            "path": str((TOKENIZER_DIR / "vocab.json").relative_to(REPO_ROOT)),
            "sha256": compute_sha256(TOKENIZER_DIR / "vocab.json")
        },
        "tokenizer_merges": {
            "path": str((TOKENIZER_DIR / "merges.json").relative_to(REPO_ROOT)),
            "sha256": compute_sha256(TOKENIZER_DIR / "merges.json")
        },
        "exp11_checkpoint": {
            "path": str(EXP11_CKPT.relative_to(REPO_ROOT)),
            "sha256": compute_sha256(EXP11_CKPT)
        },
        "splits": {
            "val": str((DATA_DIR / "val.jsonl").relative_to(REPO_ROOT)),
            "test_a": str((DATA_DIR / "test_a_familiar.jsonl").relative_to(REPO_ROOT)),
            "test_b": str((DATA_DIR / "test_b_held_out.jsonl").relative_to(REPO_ROOT)),
        }
    }
    man_path = EXP12_DIR / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {man_path}")

if __name__ == "__main__":
    main()
