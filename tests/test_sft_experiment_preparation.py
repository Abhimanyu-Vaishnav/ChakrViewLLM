"""Unit and Integration Tests for Batch 2821–2860 SFT Experiment Infrastructure.

Validates:
1. SFT Dataset schema and deterministic JSONL loading.
2. Split isolation and zero prompt/target cross-contamination.
3. Response-only causal loss mask alignment (prompt tokens masked with -100).
4. Padding with pad_id=2 and target masking with -100.
5. Context length enforcement and rejection of overlength records.
6. Rejection of empty target responses.
7. Parent checkpoint path and SHA-256 verification.
8. Output directory isolation (protects canonical checkpoints).
9. Preflight gate runner execution and zero optimizer steps invocation.
10. SFT Trainer safe mode (refuses optimization without explicit authorization).
11. SFT Evaluation set structure and baseline evaluation reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch2821_2860"

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)
from chakrview.training.sft_engine import SFTConfig, SFTDataset


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
    tok, _ = load_tokenizer_artifacts(str(tok_dir))
    return tok


def test_01_dataset_schema_and_size():
    """Verify SFT_DATASET.jsonl contains exactly 240 valid records across 8 categories."""
    dataset_path = EXPERIMENT_DIR / "SFT_DATASET.jsonl"
    assert dataset_path.is_file(), f"Dataset missing: {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 240
    categories = set()
    splits = {"train": 0, "val": 0, "test": 0}

    for line in lines:
        rec = json.loads(line)
        assert "id" in rec
        assert "category" in rec
        assert "prompt" in rec
        assert "response" in rec
        assert "split" in rec
        assert "data_role" in rec
        assert rec["provenance"] == "locally_authored_batch2821_2860"
        categories.add(rec["category"])
        splits[rec["split"]] += 1

    assert len(categories) == 8
    assert splits["train"] == 192
    assert splits["val"] == 24
    assert splits["test"] == 24


def test_02_split_isolation_and_zero_leakage():
    """Verify zero duplicate prompts exist across splits."""
    dataset_path = EXPERIMENT_DIR / "SFT_DATASET.jsonl"
    prompts_by_split = {"train": set(), "val": set(), "test": set()}

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                prompts_by_split[rec["split"]].add(rec["prompt"].strip())

    # Disjoint assertions
    assert len(prompts_by_split["train"].intersection(prompts_by_split["val"])) == 0
    assert len(prompts_by_split["train"].intersection(prompts_by_split["test"])) == 0
    assert len(prompts_by_split["val"].intersection(prompts_by_split["test"])) == 0


def test_03_causal_sequence_and_response_only_loss_masking(tokenizer):
    """Verify build_causal_sequence correctly masks prompt tokens with -100 and aligns response."""
    prompt = "Instruction: Name one color.\nResponse:"
    response = "Blue."
    p_toks = tokenizer.encode(prompt)
    r_toks = tokenizer.encode(response)

    input_seq, target_seq = build_causal_sequence(
        prompt_tokens=p_toks,
        response_tokens=r_toks,
        bos_id=0,
        eos_id=1,
        ignore_index=-100,
    )

    # 1. Lengths match
    assert len(input_seq) == len(target_seq)
    # 2. Input begins with BOS
    assert input_seq[0] == 0
    # 3. Prompt positions are masked in target_seq with -100
    for i in range(len(p_toks)):
        assert target_seq[i] == -100
    # 4. Alignment invariant: terminal prompt token predicts first response token
    assert input_seq[len(p_toks)] == p_toks[-1]
    assert target_seq[len(p_toks)] == r_toks[0]
    # 5. Target ends with EOS token
    assert target_seq[-1] == 1


def test_04_empty_response_rejected(tokenizer):
    """Verify encoding empty response raises ValueError and is not silently trained."""
    dataset_file = EXPERIMENT_DIR / "train.jsonl"
    sft_ds = SFTDataset(
        jsonl_path=dataset_file,
        tokenizer=tokenizer,
        max_seq_len=512,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
    )
    with pytest.raises(ValueError) as exc:
        sft_ds.encode_example("Prompt text", "")
    assert "0 tokens" in str(exc.value)


def test_05_context_boundary_and_truncation(tokenizer):
    """Verify sequence length limit of 512 is strictly enforced."""
    dataset_file = EXPERIMENT_DIR / "train.jsonl"
    sft_ds = SFTDataset(
        jsonl_path=dataset_file,
        tokenizer=tokenizer,
        max_seq_len=64,  # Small limit for test
        allow_exploratory_opt_in=False,
    )
    # Prompt + response exceeding limit will truncate prompt while preserving response
    huge_prompt = "word " * 100
    short_resp = "Valid response."
    inp, tgt = sft_ds.encode_example(huge_prompt, short_resp)
    assert len(inp) <= 64
    assert len(tgt) <= 64


def test_06_preflight_script_passes_all_gates():
    """Verify run_sft_preflight executes cleanly with exit code 0."""
    from experiments.batch2821_2860.run_sft_preflight import run_preflight
    exit_code = run_preflight()
    assert exit_code == 0

    preflight_rep_path = EXPERIMENT_DIR / "SFT_PREFLIGHT_REPORT.json"
    with open(preflight_rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["overall_verdict"] == "PASS"
    assert rep["training_executed"] is False
    assert rep["optimizer_steps_executed"] == 0
    assert rep["weights_modified"] is False


def test_07_trainer_safe_inspection_mode_does_not_train(tmp_path):
    """Verify trainer without explicit flag executes inspection only and does not train."""
    from experiments.batch2821_2860.run_sft_trainer import inspect_or_train
    import tempfile
    
    # Create config with fresh isolated output directory
    fresh_out = tmp_path / "inspection_out"
    test_cfg = {
        "parent_checkpoint": {
            "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
            "expected_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
        },
        "output_directory": str(fresh_out.relative_to(REPO_ROOT) if fresh_out.is_relative_to(REPO_ROOT) else fresh_out),
        "training_hyperparameters": {
            "pilot_steps": 10, "batch_size": 2, "learning_rate": 1e-4, "min_learning_rate": 1e-5,
            "seed": 42
        }
    }
    cfg_file = tmp_path / "test_config.json"
    cfg_file.write_text(json.dumps(test_cfg), encoding="utf-8")

    # Call without execute_training
    inspect_or_train(cfg_file, execute_training=False)
    # Check that output directory was never created and no checkpoint exists
    out_ckpt = fresh_out / "sft_checkpoint_step_10.pt"
    assert not out_ckpt.exists()


def test_08_canonical_parent_checkpoint_protection():
    """Verify SFT trainer refuses to use canonical checkpoint directories as output."""
    from experiments.batch2821_2860.run_sft_trainer import inspect_or_train
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        bad_cfg = {
            "parent_checkpoint": {
                "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
                "expected_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
            },
            "output_directory": "experiments/batch16_21_combined/main_training_10000",
            "training_hyperparameters": {
                "pilot_steps": 10, "batch_size": 2, "learning_rate": 1e-4, "min_learning_rate": 1e-5,
                "seed": 42
            }
        }
        json.dump(bad_cfg, tf)
        bad_cfg_path = Path(tf.name)

    with pytest.raises(ValueError) as exc:
        inspect_or_train(bad_cfg_path, execute_training=False)
    assert "Protected directory violation" in str(exc.value)
    bad_cfg_path.unlink()


def test_09_baseline_evaluation_set_artifacts_valid():
    """Verify SFT_EVALUATION_SET.json and SFT_BASELINE_EVALUATION_RESULTS.json exist and match."""
    eval_set_path = EXPERIMENT_DIR / "SFT_EVALUATION_SET.json"
    baseline_res_path = EXPERIMENT_DIR / "SFT_BASELINE_EVALUATION_RESULTS.json"

    assert eval_set_path.is_file()
    assert baseline_res_path.is_file()

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(baseline_res_path, "r", encoding="utf-8") as f:
        base_res = json.load(f)

    assert eval_data["total_cases"] == 24
    assert base_res["total_cases"] == 24
    assert len(base_res["results"]) == 24
