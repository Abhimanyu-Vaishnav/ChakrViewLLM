"""Comparative Capability Evaluation Harness for SFT V3 Pilot.

Evaluates:
1. Canonical Step-10,000 Base Checkpoint
2. Newly Trained SFT V3 Pilot Candidate Checkpoint (pilot_run_v3/sft_best_checkpoint.pt)

Across:
A. Native Unconstrained Generation (greedy, temp=0.0):
   - Valid JSON rate
   - Schema compliance rate
   - Exact task match (EM)
   - Code-attractor contamination rate (doctest / __main__)
   - Premature EOS / empty rate
   - Average token length
B. Schema-Constrained Generation:
   - Valid JSON rate
   - Schema compliance rate
   - Exact task match (EM)
   - Deadlocks and budget exhaustion
C. Replay Continuation Probes (10 validation replay cases):
   - Active continuation length
   - Empty / immediate-EOS rate
D. Language Model Validation Loss (using identical loss definition)

Saves machine-readable results to:
experiments/sft_v3_pilot_readiness/pilot_run_v3/EVALUATION_RESULTS.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts
from chakrview.training.sft_engine import SFTDataset
from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    PropertySpec,
    generate_schema_constrained,
)

BASE_CKPT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANDIDATE_CKPT_PATH = REPO_ROOT / "experiments" / "sft_v3_pilot_readiness" / "pilot_run_v3" / "sft_best_checkpoint.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
VAL_FILE = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
OUTPUT_JSON = REPO_ROOT / "experiments" / "sft_v3_pilot_readiness" / "pilot_run_v3" / "EVALUATION_RESULTS.json"


def load_model(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> ChakrMicro:
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
    )
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_validation_loss(model: ChakrMicro, val_dataset: SFTDataset, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.RandomState(42)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # Evaluate across entire dataset deterministically
        for rec in val_dataset.records:
            append_eos = rec.get("append_eos", True)
            inp, tgt = val_dataset.encode_example(rec["prompt"], rec["response"], append_eos=append_eos)
            x = torch.tensor([inp], dtype=torch.long, device=device)
            y = torch.tensor([tgt], dtype=torch.long, device=device)
            active = (y != -100).sum().item()
            if active == 0:
                continue
            logits = model(x)
            loss = loss_fn(logits.view(-1, 4096), y.view(-1))
            total_loss += loss.item() * active
            total_tokens += active

    mean_loss = total_loss / max(1, total_tokens)
    perp = math.exp(min(20.0, mean_loss))
    return {
        "val_loss": round(mean_loss, 4),
        "val_perplexity": round(perp, 2),
        "evaluated_tokens": total_tokens,
    }


def generate_unconstrained(model: ChakrMicro, tokenizer: Any, prompt_str: str, max_new_tokens: int = 48) -> str:
    input_ids = tokenizer.encode(prompt_str)
    curr_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = curr_ids[-512:]
            x = torch.tensor([idx_cond], dtype=torch.long, device=torch.device("cpu"))
            logits = model(x)
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            if next_token == 1:  # EOS
                break
            curr_ids.append(next_token)

    gen_tokens = curr_ids[len(input_ids):]
    try:
        return tokenizer.decode(gen_tokens, errors="replace")
    except TypeError:
        return tokenizer.decode(gen_tokens)


def evaluate_model_on_val(model: ChakrMicro, tokenizer: Any, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "structured_total": 0,
        "replay_total": 0,
        "native": {
            "valid_json": 0,
            "schema_compliant": 0,
            "exact_task_match": 0,
            "code_attractor_count": 0,
            "premature_eos": 0,
            "empty_output": 0,
            "total_tokens_generated": 0,
        },
        "constrained": {
            "valid_json": 0,
            "schema_compliant": 0,
            "exact_task_match": 0,
            "deadlocks": 0,
            "budget_exhaustion": 0,
        },
        "replay": {
            "total_continuation_tokens": 0,
            "immediate_eos_count": 0,
            "average_continuation_length": 0.0,
        },
        "per_category": {},
        "details": [],
    }

    for r in records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        expected = r["expected_answer"]
        req_keys = r.get("required_keys", [])
        ftypes = r.get("field_types") or {}

        if cat not in metrics["per_category"]:
            metrics["per_category"][cat] = {
                "total": 0,
                "native_valid": 0,
                "native_em": 0,
                "constrained_valid": 0,
                "constrained_em": 0,
            }
        metrics["per_category"][cat]["total"] += 1

        if cat == "base_continuation_replay":
            metrics["replay_total"] += 1
            gen_text = generate_unconstrained(model, tokenizer, prompt, max_new_tokens=48)
            gen_toks = tokenizer.encode(gen_text)
            tok_len = len(gen_toks)
            metrics["replay"]["total_continuation_tokens"] += tok_len
            if tok_len <= 1:
                metrics["replay"]["immediate_eos_count"] += 1
            metrics["details"].append({
                "id": rid,
                "category": cat,
                "generated_text": gen_text,
                "tokens_generated": tok_len,
            })
            continue

        metrics["structured_total"] += 1

        # A. Native Unconstrained
        nat_out = generate_unconstrained(model, tokenizer, prompt, max_new_tokens=48)
        nat_toks = tokenizer.encode(nat_out)
        metrics["native"]["total_tokens_generated"] += len(nat_toks)

        if len(nat_toks) == 0:
            metrics["native"]["empty_output"] += 1
            metrics["native"]["premature_eos"] += 1

        is_code_attractor = ("__main__" in nat_out or "import doctest" in nat_out or "def " in nat_out)
        if is_code_attractor:
            metrics["native"]["code_attractor_count"] += 1

        nat_valid = False
        nat_schema = False
        nat_em = False
        try:
            parsed = json.loads(nat_out.strip())
            nat_valid = True
            metrics["native"]["valid_json"] += 1
            metrics["per_category"][cat]["native_valid"] += 1
            if isinstance(parsed, dict) and all(k in parsed for k in req_keys):
                nat_schema = True
                metrics["native"]["schema_compliant"] += 1
            if parsed == expected:
                nat_em = True
                metrics["native"]["exact_task_match"] += 1
                metrics["per_category"][cat]["native_em"] += 1
        except Exception:
            pass

        # B. Constrained Generation
        prop_specs = []
        for k in req_keys:
            t = ftypes.get(k, "string")
            if t == "integer":
                jtype = JSONType.INTEGER
            elif t == "number":
                jtype = JSONType.NUMBER
            elif t == "boolean":
                jtype = JSONType.BOOLEAN
            elif t == "null":
                jtype = JSONType.NULL
            else:
                jtype = JSONType.STRING
            prop_specs.append(PropertySpec(name=k, allowed_types={jtype}, required=True))

        schema = JSONSchema(
            properties={p.name: p for p in prop_specs},
            allow_additional_properties=False,
            allow_duplicate_keys=False,
        )
        prompt_tokens = tokenizer.encode(prompt)

        dec_res = generate_schema_constrained(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            schema=schema,
            max_new_tokens=64,
            strict_spacing=True,
        )

        const_out = dec_res.get("generated_text", "")
        stop_reason = dec_res.get("stop_reason", "")
        parsed_c = dec_res.get("parsed_object")
        const_valid = (parsed_c is not None)
        const_schema = False
        const_em = False

        if stop_reason == "DEADLOCK":
            metrics["constrained"]["deadlocks"] += 1
        elif stop_reason in ("MAX_TOKENS", "CONTEXT_LIMIT"):
            metrics["constrained"]["budget_exhaustion"] += 1

        if const_valid:
            metrics["constrained"]["valid_json"] += 1
            metrics["per_category"][cat]["constrained_valid"] += 1
            if isinstance(parsed_c, dict) and all(k in parsed_c for k in req_keys):
                const_schema = True
                metrics["constrained"]["schema_compliant"] += 1
            if parsed_c == expected:
                const_em = True
                metrics["constrained"]["exact_task_match"] += 1
                metrics["per_category"][cat]["constrained_em"] += 1

        metrics["details"].append({
            "id": rid,
            "category": cat,
            "native_output": nat_out,
            "native_valid": nat_valid,
            "native_schema": nat_schema,
            "native_em": nat_em,
            "is_code_attractor": is_code_attractor,
            "constrained_output": const_out,
            "constrained_valid": const_valid,
            "constrained_schema": const_schema,
            "constrained_em": const_em,
            "constrained_stop_reason": stop_reason,
        })

    if metrics["replay_total"] > 0:
        metrics["replay"]["average_continuation_length"] = round(
            metrics["replay"]["total_continuation_tokens"] / metrics["replay_total"], 2
        )
    return metrics


def run_comparative_evaluation():
    device = torch.device("cpu")
    tokenizer = load_experiment_artifacts(TOKENIZER_DIR)

    with open(VAL_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    val_dataset = SFTDataset(VAL_FILE, tokenizer, max_seq_len=512, is_training=False, strip_prompt_delimiters=False)

    print("=================================================================")
    print("CHAKRVIEW SFT V3 COMPARATIVE EVALUATION (BASE VS PILOT CANDIDATE)")
    print("=================================================================")

    # 1. Base Model Evaluation
    print("Evaluating Canonical Step-10,000 Base Checkpoint...")
    base_model = load_model(BASE_CKPT_PATH, device)
    base_loss_metrics = compute_validation_loss(base_model, val_dataset, device)
    base_eval = evaluate_model_on_val(base_model, tokenizer, records)
    del base_model

    # 2. Candidate Model Evaluation
    print("Evaluating Newly Trained SFT V3 Candidate Checkpoint...")
    cand_model = load_model(CANDIDATE_CKPT_PATH, device)
    cand_loss_metrics = compute_validation_loss(cand_model, val_dataset, device)
    cand_eval = evaluate_model_on_val(cand_model, tokenizer, records)
    del cand_model

    comparative_results = {
        "evaluation_scope": "SFT V3 40-record validation split (30 structured, 10 replay)",
        "base_checkpoint": {
            "path": str(BASE_CKPT_PATH),
            "loss_metrics": base_loss_metrics,
            "evaluation": base_eval,
        },
        "candidate_checkpoint": {
            "path": str(CANDIDATE_CKPT_PATH),
            "loss_metrics": cand_loss_metrics,
            "evaluation": cand_eval,
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2)

    print(f"\nEvaluation results written to: {OUTPUT_JSON}")
    print("=================================================================")
    print("COMPARATIVE EVALUATION SUMMARY:")
    print("=================================================================")
    print(f"Validation Loss:         Base: {base_loss_metrics['val_loss']}  --> Candidate: {cand_loss_metrics['val_loss']}")
    print(f"Validation Perplexity:   Base: {base_loss_metrics['val_perplexity']}  --> Candidate: {cand_loss_metrics['val_perplexity']}")
    print(f"Native Valid JSON:       Base: {base_eval['native']['valid_json']}/30  --> Candidate: {cand_eval['native']['valid_json']}/30")
    print(f"Native Exact Match:      Base: {base_eval['native']['exact_task_match']}/30  --> Candidate: {cand_eval['native']['exact_task_match']}/30")
    print(f"Native Code Attractor:   Base: {base_eval['native']['code_attractor_count']}/30  --> Candidate: {cand_eval['native']['code_attractor_count']}/30")
    print(f"Constrained Valid JSON:  Base: {base_eval['constrained']['valid_json']}/30  --> Candidate: {cand_eval['constrained']['valid_json']}/30")
    print(f"Constrained Exact Match: Base: {base_eval['constrained']['exact_task_match']}/30  --> Candidate: {cand_eval['constrained']['exact_task_match']}/30")
    print(f"Replay Avg Cont. Length: Base: {base_eval['replay']['average_continuation_length']}  --> Candidate: {cand_eval['replay']['average_continuation_length']}")
    print(f"Replay Imm. EOS Count:   Base: {base_eval['replay']['immediate_eos_count']}/10  --> Candidate: {cand_eval['replay']['immediate_eos_count']}/10")
    print("=================================================================")


if __name__ == "__main__":
    run_comparative_evaluation()
