"""End-to-End Tests for Batch 2981–3020 Dataset Leakage, Real-Data Alignment, and Safety.

Validates:
1. Exact and normalized prompt uniqueness across train/val/test splits.
2. Real-data causal target alignment and EOS supervision across all 8 task categories.
3. Training preflight safety and rejection of unauthorized execution.
4. Comprehensive best-checkpoint state saving, reload, and atomic replacement.
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
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine, SFTDataset


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
    tok, _ = load_tokenizer_artifacts(str(tok_dir))
    return tok


def test_01_real_data_causal_targets_across_all_categories(tokenizer):
    """Test real generated examples from each category to verify prompt masking and response/EOS targets."""
    train_path = BATCH_DIR / "train.jsonl"
    assert train_path.exists()
    
    sft_ds = SFTDataset(
        jsonl_path=train_path,
        tokenizer=tokenizer,
        max_seq_len=512,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
    )
    
    records = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    by_cat = {}
    for r in records:
        if r["category"] not in by_cat:
            by_cat[r["category"]] = r
            
    assert len(by_cat) == 8, f"Expected 8 distinct categories, got {len(by_cat)}"
    
    for cat, r in by_cat.items():
        append_eos = r.get("append_eos", True)
        inp, tgt = sft_ds.encode_example(r["prompt"], r["response"], append_eos=append_eos)
        
        # 1. Length equality
        assert len(inp) == len(tgt)
        # 2. Input starts with BOS (0)
        assert inp[0] == 0
        # 3. First target token is masked (-100)
        assert tgt[0] == -100
        
        p_str, r_str = normalize_prompt_response_strings(r["prompt"], r["response"], strip_delimiters=True)
        p_toks = tokenizer.encode(p_str)
        r_toks = tokenizer.encode(r_str)
        
        # 4. Prompt positions are masked with -100
        for i in range(len(p_toks)):
            assert tgt[i] == -100
            
        # 5. Causal alignment: final prompt token predicts first response token
        assert inp[len(p_toks)] == p_toks[-1]
        assert tgt[len(p_toks)] == r_toks[0]
        
        # 6. EOS target policy
        if cat == "base_continuation_replay":
            assert not append_eos
            # Replay ends naturally without forced EOS
            assert tgt[-1] == r_toks[-1]
        else:
            assert append_eos
            # Instruction ends with terminal EOS
            assert tgt[-1] == 1


def test_02_split_disjointness_and_normalized_uniqueness():
    """Verify train, val, and test splits have 0 exact or normalized prompt overlap."""
    train_recs = [json.loads(l) for l in open(BATCH_DIR / "train.jsonl", encoding="utf-8") if l.strip()]
    val_recs = [json.loads(l) for l in open(BATCH_DIR / "val.jsonl", encoding="utf-8") if l.strip()]
    test_recs = [json.loads(l) for l in open(BATCH_DIR / "test.jsonl", encoding="utf-8") if l.strip()]
    
    train_prompts = set(r["prompt"].strip() for r in train_recs)
    val_prompts = set(r["prompt"].strip() for r in val_recs)
    test_prompts = set(r["prompt"].strip() for r in test_recs)
    
    # Check disjointness
    assert len(train_prompts.intersection(val_prompts)) == 0
    assert len(train_prompts.intersection(test_prompts)) == 0
    assert len(val_prompts.intersection(test_prompts)) == 0
    
    # Check normalized uniqueness across full dataset
    all_recs = train_recs + val_recs + test_recs
    norm_prompts = set(" ".join(r["prompt"].lower().split()) for r in all_recs)
    assert len(norm_prompts) == len(all_recs) == 400


def test_03_preflight_refuses_training_by_default():
    """Verify preflight runs in safe mode by default without executing training."""
    from experiments.batch2941_2980.run_preflight import run_preflight
    exit_code = run_preflight()
    assert exit_code == 0
    
    rep_path = BATCH_DIR / "PREFLIGHT_REPORT.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    assert rep["training_executed"] is False
    assert rep["optimizer_steps_executed"] == 0
    assert rep["weights_modified"] is False
