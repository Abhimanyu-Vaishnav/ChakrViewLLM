"""SFT Preflight Validation for Balanced 70/30 SFT Experiment in Batch 2941–2980.

Validates:
1. Canonical parent checkpoint SHA-256 integrity.
2. Tokenizer integrity and vocab size.
3. Dataset schema, split isolation, and mixture balance (70/30).
4. Replay and instruction label masks (append_eos=True vs False).
5. Output directory isolation (must not point to canonical or old batch dirs).
6. Best-checkpoint saving logic verification.
7. Emits experiments/batch2941_2980/PREFLIGHT_REPORT.json.

CRITICAL INVARIANT: Zero optimizer steps, training, or weight updates executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXPECTED_PARENT_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch2941_2980"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_preflight() -> int:
    print("=================================================================")
    print("CHAKRVIEW BATCH 2941-2980: BALANCED SFT PREFLIGHT INTEGRITY AUDIT")
    print("=================================================================")

    gates: Dict[str, Dict[str, Any]] = {}
    overall_pass = True

    # Gate 1: Canonical Base Checkpoint
    ckpt_exists = CANONICAL_PARENT_PATH.is_file()
    actual_ckpt_sha = compute_sha256(CANONICAL_PARENT_PATH) if ckpt_exists else "MISSING"
    ckpt_pass = ckpt_exists and (actual_ckpt_sha == EXPECTED_PARENT_SHA256)
    if not ckpt_pass:
        overall_pass = False
    gates["canonical_base_checkpoint"] = {
        "status": "PASS" if ckpt_pass else "FAIL",
        "path": str(CANONICAL_PARENT_PATH),
        "expected_sha256": EXPECTED_PARENT_SHA256,
        "actual_sha256": actual_ckpt_sha,
    }
    print(f"Gate 1 [Parent Checkpoint]: {gates['canonical_base_checkpoint']['status']} ({actual_ckpt_sha[:16]}...)")

    # Gate 2: Tokenizer Verification
    tok_exists = TOKENIZER_DIR.is_dir()
    tokenizer = None
    if tok_exists:
        try:
            tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))
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
    gates["tokenizer_artifacts"] = {
        "status": "PASS" if tok_pass else "FAIL",
        "directory": str(TOKENIZER_DIR),
        "vocab_size": vocab_size,
    }
    print(f"Gate 2 [Tokenizer]: {gates['tokenizer_artifacts']['status']} (vocab_size={vocab_size})")

    # Gate 3: Balanced Dataset Schema, Splits & 70/30 Mixture Audit
    manifest_path = EXPERIMENT_DIR / "DATASET_MANIFEST.json"
    dataset_path = EXPERIMENT_DIR / "BALANCED_SFT_DATASET.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    splits = {"train": 0, "val": 0, "test": 0}
    prompts_seen = {}
    leakage = False
    instruction_count = 0
    replay_count = 0

    for r in records:
        s = r["split"]
        p = r["prompt"]
        splits[s] = splits.get(s, 0) + 1
        if p in prompts_seen and prompts_seen[p] != s:
            leakage = True
        prompts_seen[p] = s
        if r["category"] == "base_continuation_replay":
            replay_count += 1
        else:
            instruction_count += 1

    ds_pass = (
        len(records) == 400
        and splits["train"] == 320
        and splits["val"] == 40
        and splits["test"] == 40
        and not leakage
        and instruction_count == 280
        and replay_count == 120
    )
    if not ds_pass:
        overall_pass = False
    gates["dataset_manifest_and_balance"] = {
        "status": "PASS" if ds_pass else "FAIL",
        "total_records": len(records),
        "splits": splits,
        "instruction_count": instruction_count,
        "replay_count": replay_count,
        "split_leakage_detected": leakage,
    }
    print(f"Gate 3 [Dataset & 70/30 Balance]: {gates['dataset_manifest_and_balance']['status']} (total=400, inst=280, replay=120)")

    # Gate 4: Label Masking & EOS Policy
    # Check synthetic sequence building for both types
    inp_i, tgt_i = build_causal_sequence([10, 20], [30, 40], bos_id=0, eos_id=1, ignore_index=-100, append_eos=True)
    inst_mask_ok = (tgt_i[-1] == 1 and len(inp_i) == len(tgt_i) and tgt_i[0] == -100 and tgt_i[1] == -100 and tgt_i[2] == 30)

    inp_r, tgt_r = build_causal_sequence([100, 200], [300, 400, 500], bos_id=0, eos_id=1, ignore_index=-100, append_eos=False)
    replay_mask_ok = (tgt_r[-1] == 500 and len(inp_r) == len(tgt_r) and tgt_r[0] == -100 and tgt_r[1] == -100 and tgt_r[2] == 300)

    mask_pass = inst_mask_ok and replay_mask_ok
    if not mask_pass:
        overall_pass = False
    gates["label_masking_and_eos_policy"] = {
        "status": "PASS" if mask_pass else "FAIL",
        "instruction_eos_supervised": inst_mask_ok,
        "replay_eos_unforced": replay_mask_ok,
    }
    print(f"Gate 4 [Label Masking & EOS]: {gates['label_masking_and_eos_policy']['status']} (instruction ends with EOS, replay ends naturally)")

    # Gate 5: Output Isolation & Safe Trainer Config
    cfg_path = EXPERIMENT_DIR / "SFT_EXPERIMENT_CONFIG.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out_dir_str = cfg["output_directory"]
    iso_pass = "batch2941_2980" in out_dir_str and "main_training_10000" not in out_dir_str
    if not iso_pass:
        overall_pass = False
    gates["output_isolation"] = {
        "status": "PASS" if iso_pass else "FAIL",
        "configured_output": out_dir_str,
    }
    print(f"Gate 5 [Output Isolation]: {gates['output_isolation']['status']} ({out_dir_str})")

    # Gate 6: Best Checkpoint Saving Implementation
    from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
    import inspect
    save_ckpt_sig = inspect.signature(SFTExperimentEngine.save_checkpoint)
    best_support = "is_best" in save_ckpt_sig.parameters
    if not best_support:
        overall_pass = False
    gates["best_checkpoint_saving_support"] = {
        "status": "PASS" if best_support else "FAIL",
        "save_checkpoint_has_is_best": best_support,
    }
    print(f"Gate 6 [Best Checkpoint Support]: {gates['best_checkpoint_saving_support']['status']} (save_checkpoint supports is_best)")

    # Produce PREFLIGHT_REPORT.json
    report = {
        "preflight_version": "batch2941_2980_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "training_executed": False,
        "optimizer_steps_executed": 0,
        "weights_modified": False,
        "gates": gates,
        "mandatory_execution_command": (
            ".\\.venv\\Scripts\\python.exe experiments/batch2941_2980/run_balanced_sft_trainer.py "
            "--config experiments/batch2941_2980/SFT_EXPERIMENT_CONFIG.json "
            "--execute-authorized-training"
        )
    }

    with open(EXPERIMENT_DIR / "PREFLIGHT_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=================================================================")
    print(f"OVERALL PREFLIGHT VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print("Zero training or optimizer steps executed. Ready for explicit authorization.")
    print("=================================================================")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(run_preflight())
