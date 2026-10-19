"""SFT Preflight Validation & Dataset Inspection for Batch 2821–2860.

Validates all pre-conditions without training or invoking optimizer steps:
1. Parent checkpoint existence and SHA-256 integrity.
2. Tokenizer integrity and vocabulary size.
3. Dataset schema, split isolation, duplicate leakage detection.
4. Tokenization length distribution and truncation/rejection rates.
5. Causal loss mask alignment and response-only gradient verification.
6. Output directory isolation (must not overwrite canonical artifacts).
7. Writes SFT_DATASET_REPORT.json and SFT_PREFLIGHT_REPORT.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXPECTED_PARENT_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch2821_2860"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_preflight() -> int:
    print("=================================================================")
    print("CHAKRVIEW BATCH 2821-2860: SFT PREFLIGHT INTEGRITY AUDIT")
    print("=================================================================")

    gates: Dict[str, Dict[str, Any]] = {}
    overall_pass = True

    # Gate 1: Parent Checkpoint Verification
    t0 = time.time()
    ckpt_exists = CANONICAL_PARENT_PATH.is_file()
    actual_ckpt_sha = compute_sha256(CANONICAL_PARENT_PATH) if ckpt_exists else "MISSING"
    ckpt_pass = ckpt_exists and (actual_ckpt_sha == EXPECTED_PARENT_SHA256)
    if not ckpt_pass:
        overall_pass = False
    gates["parent_checkpoint_integrity"] = {
        "status": "PASS" if ckpt_pass else "FAIL",
        "path": str(CANONICAL_PARENT_PATH),
        "expected_sha256": EXPECTED_PARENT_SHA256,
        "actual_sha256": actual_ckpt_sha,
    }
    print(f"Gate 1 [Parent Checkpoint]: {gates['parent_checkpoint_integrity']['status']} ({actual_ckpt_sha[:16]}...)")

    # Gate 2: Tokenizer Verification
    tok_exists = TOKENIZER_DIR.is_dir()
    tokenizer = None
    if tok_exists:
        try:
            tokenizer, meta = load_tokenizer_artifacts(str(TOKENIZER_DIR))
            vocab_size = tokenizer.vocab_size
            tok_pass = vocab_size == 4096
        except Exception as e:
            tok_pass = False
            vocab_size = str(e)
    else:
        tok_pass = False
        vocab_size = 0
    if not tok_pass:
        overall_pass = False
    gates["tokenizer_integrity"] = {
        "status": "PASS" if tok_pass else "FAIL",
        "directory": str(TOKENIZER_DIR),
        "vocab_size": vocab_size,
    }
    print(f"Gate 2 [Tokenizer]: {gates['tokenizer_integrity']['status']} (vocab_size={vocab_size})")

    # Gate 3: Dataset Schema, Splits, and Leakage Audit
    dataset_file = EXPERIMENT_DIR / "SFT_DATASET.jsonl"
    records: List[Dict[str, Any]] = []
    if dataset_file.is_file():
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    category_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    prompts_seen: Dict[str, str] = {}
    leakage_detected = False
    schema_valid = True

    for r in records:
        rid = r.get("id")
        cat = r.get("category")
        p = r.get("prompt")
        resp = r.get("response")
        split = r.get("split")
        role = r.get("data_role")

        if not (rid and cat and p and resp and split and role):
            schema_valid = False

        category_counts[cat] = category_counts.get(cat, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1

        # Check exact prompt leakage across splits
        if p in prompts_seen and prompts_seen[p] != split:
            leakage_detected = True
        prompts_seen[p] = split

    dataset_pass = (
        len(records) == 240
        and schema_valid
        and not leakage_detected
        and split_counts.get("train", 0) == 192
        and split_counts.get("val", 0) == 24
        and split_counts.get("test", 0) == 24
    )
    if not dataset_pass:
        overall_pass = False

    gates["dataset_schema_and_isolation"] = {
        "status": "PASS" if dataset_pass else "FAIL",
        "total_examples": len(records),
        "split_counts": split_counts,
        "category_counts": category_counts,
        "leakage_detected": leakage_detected,
    }
    print(f"Gate 3 [Dataset Schema & Isolation]: {gates['dataset_schema_and_isolation']['status']} (total={len(records)}, train={split_counts.get('train',0)}, val={split_counts.get('val',0)}, test={split_counts.get('test',0)})")

    # Gate 4: Token Length Distribution & Context Limits
    p_lengths: List[int] = []
    r_lengths: List[int] = []
    total_lengths: List[int] = []
    truncated_count = 0
    rejected_count = 0

    assert tokenizer is not None
    sample_mask_debug: List[Dict[str, Any]] = []

    for i, r in enumerate(records):
        p_str, r_str = normalize_prompt_response_strings(r["prompt"], r["response"], strip_delimiters=False)
        p_toks = tokenizer.encode(p_str)
        r_toks = tokenizer.encode(r_str)

        if not r_toks:
            rejected_count += 1
            continue

        p_len = len(p_toks)
        r_len = len(r_toks)
        tot = 1 + p_len + r_len  # BOS + prompt + response

        if tot > 512:
            truncated_count += 1

        p_lengths.append(p_len)
        r_lengths.append(r_len)
        total_lengths.append(tot)

        # Capture first 2 examples for loss mask position audit
        if i < 2:
            inp_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
            sample_mask_debug.append({
                "record_id": r["id"],
                "prompt_token_count": p_len,
                "response_token_count": r_len,
                "input_seq_len": len(inp_seq),
                "target_seq_len": len(tgt_seq),
                "masked_tokens_count": sum(1 for t in tgt_seq if t == -100),
                "active_tokens_count": sum(1 for t in tgt_seq if t != -100),
                "first_active_target": tgt_seq[p_len],
                "expected_first_response_token": r_toks[0],
                "alignment_verified": tgt_seq[p_len] == r_toks[0]
            })

    length_pass = (len(total_lengths) == len(records)) and (max(total_lengths) <= 512) and (truncated_count == 0) and (rejected_count == 0)
    if not length_pass:
        overall_pass = False

    gates["token_length_and_context_boundary"] = {
        "status": "PASS" if length_pass else "FAIL",
        "max_context_limit": 512,
        "max_observed_seq_len": max(total_lengths) if total_lengths else 0,
        "min_observed_seq_len": min(total_lengths) if total_lengths else 0,
        "mean_prompt_tokens": round(sum(p_lengths) / len(p_lengths), 2) if p_lengths else 0,
        "mean_response_tokens": round(sum(r_lengths) / len(r_lengths), 2) if r_lengths else 0,
        "mean_total_tokens": round(sum(total_lengths) / len(total_lengths), 2) if total_lengths else 0,
        "truncated_examples": truncated_count,
        "rejected_examples": rejected_count,
    }
    print(f"Gate 4 [Context Boundary]: {gates['token_length_and_context_boundary']['status']} (max_len={max(total_lengths) if total_lengths else 0}/512, truncated={truncated_count})")

    # Gate 5: Causal Loss Mask Alignment
    mask_pass = all(s["alignment_verified"] and s["active_tokens_count"] == s["response_token_count"] + 1 for s in sample_mask_debug)
    if not mask_pass:
        overall_pass = False
    gates["causal_loss_masking_alignment"] = {
        "status": "PASS" if mask_pass else "FAIL",
        "sample_verifications": sample_mask_debug,
    }
    print(f"Gate 5 [Loss Mask Alignment]: {gates['causal_loss_masking_alignment']['status']} (prompt tokens masked with -100, response target verified)")

    # Gate 6: Output Directory Isolation
    config_file = EXPERIMENT_DIR / "SFT_TRAINING_CONFIG.json"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    out_dir_str = cfg_data.get("output_directory", "")
    out_dir = REPO_ROOT / out_dir_str
    # Output dir must be inside experiments/batch2821_2860 and NOT touch canonical checkpoints
    is_isolated = "batch2821_2860" in out_dir_str and "main_training_10000" not in out_dir_str
    if not is_isolated:
        overall_pass = False
    gates["output_isolation"] = {
        "status": "PASS" if is_isolated else "FAIL",
        "configured_output_dir": str(out_dir),
    }
    print(f"Gate 6 [Output Isolation]: {gates['output_isolation']['status']} ({out_dir_str})")

    # Write SFT_DATASET_REPORT.json
    dataset_report = {
        "dataset_name": "batch2821_2860_curated_sft_starter",
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_records": len(records),
        "split_counts": split_counts,
        "category_counts": category_counts,
        "token_statistics": {
            "prompt_tokens": {
                "min": min(p_lengths) if p_lengths else 0,
                "max": max(p_lengths) if p_lengths else 0,
                "mean": round(sum(p_lengths) / len(p_lengths), 2) if p_lengths else 0,
            },
            "response_tokens": {
                "min": min(r_lengths) if r_lengths else 0,
                "max": max(r_lengths) if r_lengths else 0,
                "mean": round(sum(r_lengths) / len(r_lengths), 2) if r_lengths else 0,
            },
            "total_sequence_tokens": {
                "min": min(total_lengths) if total_lengths else 0,
                "max": max(total_lengths) if total_lengths else 0,
                "mean": round(sum(total_lengths) / len(total_lengths), 2) if total_lengths else 0,
            },
            "truncated_count": truncated_count,
            "rejected_count": rejected_count,
        },
        "leakage_check_passed": not leakage_detected,
        "schema_check_passed": schema_valid,
        "provenance_distribution": {
            "locally_authored_batch2821_2860": len(records)
        }
    }
    with open(EXPERIMENT_DIR / "SFT_DATASET_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(dataset_report, f, indent=2)

    # Write SFT_PREFLIGHT_REPORT.json
    preflight_report = {
        "preflight_version": "batch2821_2860_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "training_executed": False,
        "optimizer_steps_executed": 0,
        "weights_modified": False,
        "gates": gates,
        "authorized_training_command": (
            ".\\.venv\\Scripts\\python.exe experiments/batch2821_2860/run_sft_trainer.py "
            "--config experiments/batch2821_2860/SFT_TRAINING_CONFIG.json "
            "--execute-authorized-training"
        )
    }
    with open(EXPERIMENT_DIR / "SFT_PREFLIGHT_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(preflight_report, f, indent=2)

    print("=================================================================")
    print(f"OVERALL SFT PREFLIGHT VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print("Zero training or optimizer steps executed. Ready for explicit authorization.")
    print("=================================================================")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(run_preflight())
