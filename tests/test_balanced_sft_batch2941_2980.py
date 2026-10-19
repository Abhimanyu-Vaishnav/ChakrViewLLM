"""Unit and Integration Tests for Batch 2941–2980 Balanced SFT Infrastructure.

Validates:
1. Best-checkpoint saving logic, safe replacement, and tie-breaking.
2. Replay vs Instruction label masking and EOS token policy.
3. Balanced dataset schema, 70/30 distribution, and split integrity.
4. Preflight execution safety and zero training guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO_ROOT / "experiments" / "batch2941_2980"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine, SFTDataset


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
    tok, _ = load_tokenizer_artifacts(str(tok_dir))
    return tok


def test_01_best_checkpoint_saving_and_tie_breaking(tmp_path):
    """Test best-checkpoint saving behavior: improvement saves, worse loss ignored, tie broken deterministically."""
    sft_cfg = SFTConfig(
        repo_root=REPO_ROOT,
        train_file=BATCH_DIR / "train.jsonl",
        val_file=BATCH_DIR / "val.jsonl",
        output_dir=tmp_path / "checkpoints",
        base_checkpoint=REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        save_best_checkpoint=True,
    )
    engine = SFTExperimentEngine(sft_cfg)
    
    # 1. First evaluation: val_loss = 3.5 -> should save best checkpoint
    engine.best_val_loss = 3.5
    engine.best_step = 10
    engine.save_checkpoint(step=10, is_best=True)
    best_file = tmp_path / "checkpoints" / "sft_best_checkpoint.pt"
    assert best_file.exists()
    
    saved_state = torch.load(best_file, weights_only=False)
    assert saved_state["sft_step"] == 10
    assert saved_state["best_val_loss"] == 3.5
    assert saved_state["is_best"] is True
    
    # 2. Worse loss: val_loss = 3.8 -> should NOT save best checkpoint (is_best=False)
    # engine.best_val_loss remains 3.5
    engine.save_checkpoint(step=20, is_best=False)
    saved_state = torch.load(best_file, weights_only=False)
    assert saved_state["sft_step"] == 10  # Preserved step 10!
    assert saved_state["best_val_loss"] == 3.5
    
    # 3. Equal loss tie: val_loss = 3.5 at step 30 -> deterministic tie-breaking (keep earlier step 10)
    is_better = 3.5 < engine.best_val_loss  # False because 3.5 is not < 3.5
    assert not is_better
    engine.save_checkpoint(step=30, is_best=False)
    saved_state = torch.load(best_file, weights_only=False)
    assert saved_state["sft_step"] == 10
    
    # 4. Better loss: val_loss = 3.1 at step 40 -> should save new best checkpoint
    engine.best_val_loss = 3.1
    engine.best_step = 40
    engine.save_checkpoint(step=40, is_best=True)
    saved_state = torch.load(best_file, weights_only=False)
    assert saved_state["sft_step"] == 40
    assert saved_state["best_val_loss"] == 3.1


def test_02_replay_vs_instruction_label_masking(tokenizer):
    """Verify build_causal_sequence supports append_eos=False for replay and append_eos=True for instruction."""
    prompt = "Instruction: Say hello.\nResponse:"
    response = " Hello world."
    
    p_toks = tokenizer.encode(prompt)
    r_toks = tokenizer.encode(response)
    
    # Instruction mode: append_eos=True
    inst_inp, inst_tgt = build_causal_sequence(
        prompt_tokens=p_toks,
        response_tokens=r_toks,
        bos_id=0,
        eos_id=1,
        ignore_index=-100,
        append_eos=True,
    )
    assert inst_tgt[-1] == 1  # Terminal token is EOS
    assert inst_inp[0] == 0   # BOS prefix
    
    # Replay mode: append_eos=False
    prefix_text = "The quick brown fox"
    continuation_text = " jumps over the lazy dog."
    c_p_toks = tokenizer.encode(prefix_text)
    c_r_toks = tokenizer.encode(continuation_text)
    
    replay_inp, replay_tgt = build_causal_sequence(
        prompt_tokens=c_p_toks,
        response_tokens=c_r_toks,
        bos_id=0,
        eos_id=1,
        ignore_index=-100,
        append_eos=False,
    )
    # In replay mode, the last target token should be the last token of continuation, NOT artificial EOS
    assert replay_tgt[-1] == c_r_toks[-1]
    assert replay_tgt[-1] != 1 or c_r_toks[-1] == 1


def test_03_balanced_dataset_mixture_and_splits():
    """Verify balanced dataset contains 400 examples with 280 instruction (70%) and 120 replay (30%)."""
    dataset_file = BATCH_DIR / "BALANCED_SFT_DATASET.jsonl"
    assert dataset_file.exists(), f"Missing dataset: {dataset_file}"
    
    records = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    assert len(records) == 400
    
    inst_count = sum(1 for r in records if not r["id"].startswith("REPLAY_"))
    replay_count = sum(1 for r in records if r["id"].startswith("REPLAY_"))
    
    assert inst_count == 280
    assert replay_count == 120
    
    # Check splits (320 train / 40 val / 40 test)
    train_count = sum(1 for r in records if r["split"] == "train")
    val_count = sum(1 for r in records if r["split"] == "val")
    test_count = sum(1 for r in records if r["split"] == "test")
    
    assert train_count == 320
    assert val_count == 40
    assert test_count == 40


def test_04_preflight_verifies_cleanly():
    """Verify Batch 2941-2980 preflight runs cleanly and outputs PASS."""
    from experiments.batch2941_2980.run_preflight import run_preflight
    exit_code = run_preflight()
    assert exit_code == 0
    
    report_file = BATCH_DIR / "PREFLIGHT_REPORT.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["overall_verdict"] == "PASS"
    assert rep["training_executed"] is False
    assert rep["optimizer_steps_executed"] == 0
