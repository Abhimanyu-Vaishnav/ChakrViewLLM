"""
evaluate_exp09_comparison.py
ChakrView Experiment 09 — Comprehensive Evaluation and Comparison Harness

Evaluates across:
1. Validation Set (228 examples)
2. Test A: Familiar Token Identities in Novel Contexts (214 examples)
3. Test B: Held-Out Token Identities (300 examples)

Models compared:
- Baseline Parent (ChakrMicro Step-10000)
- Arm A Best Checkpoint (Broader-Vocabulary Causal SFT)
- Arm B Best Checkpoint (Isolated Pointer Head)

Tracks:
- Exact copying count and percentage (Token EM).
- Teacher-forced first-token accuracy (Arm A & Parent).
- Pointer source-position accuracy (Arm B).
- Free-generation exact match (Arm A & Parent).
- Distractor lookup subset accuracy.
- Output diversity and repetition collapse.
- Parameter count and memory.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp09_vocab_vs_pointer.pointer_head import ChakrPointerModel

EXP09_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP09_DIR / "data"
CHECKPOINT_DIR = EXP09_DIR / "checkpoints"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def eval_causal_split(model: ChakrMicro, tokenizer, split_path: Path, device="cpu"):
    with open(split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    first_token_hits = 0
    free_gen_hits = 0
    ranks = []
    probs = []
    predictions = []

    distractor_total = 0
    distractor_hits = 0

    with torch.no_grad():
        for rec in records:
            tmpl = rec.get("template_type", "")
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]

            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)
            full_seq = [0] + p_toks + r_toks + [1]
            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)

            logits = model(inp_tensor)
            target_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(target_logit).item())
            softmax_probs = F.softmax(target_logit, dim=-1)
            target_prob = softmax_probs[target_tok_id].item()
            sorted_indices = torch.argsort(target_logit, descending=True)
            target_rank = int((sorted_indices == target_tok_id).nonzero(as_tuple=True)[0].item()) + 1

            ranks.append(target_rank)
            probs.append(target_prob)
            if pred_tok_id == target_tok_id:
                first_token_hits += 1

            # Free generation
            curr = [0] + list(p_toks)
            gen_tokens = []
            for _ in range(8):
                g_inp = torch.tensor([curr], dtype=torch.long, device=device)
                g_logits = model(g_inp)
                nxt = int(torch.argmax(g_logits[0, -1, :]).item())
                if nxt == 1:
                    break
                gen_tokens.append(nxt)
                curr.append(nxt)

            try:
                gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
            except Exception:
                gen_text = tokenizer.decode(gen_tokens).strip()

            is_em = (gen_text == target_val.strip())
            if is_em:
                free_gen_hits += 1

            if tmpl == "multi_item_lookup":
                distractor_total += 1
                if is_em:
                    distractor_hits += 1

            predictions.append({
                "prompt": prompt,
                "expected": target_val,
                "generated": gen_text,
                "target_rank": target_rank,
                "target_prob": target_prob,
                "exact_match": is_em
            })

    unique_outputs = set(p["generated"] for p in predictions)
    return {
        "total": total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "mean_rank": float(sum(ranks) / total),
        "mean_prob": float(sum(probs) / total),
        "distractor_total": distractor_total,
        "distractor_hits": distractor_hits,
        "distractor_em": (distractor_hits / distractor_total * 100.0) if distractor_total > 0 else 0.0,
        "output_diversity": len(unique_outputs) / total if total > 0 else 0.0,
        "sample_predictions": predictions[:5]
    }

def eval_pointer_split_full(model: ChakrPointerModel, tokenizer, split_path: Path, device="cpu"):
    with open(split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    pos_hits = 0
    token_em_hits = 0
    predictions = []

    distractor_total = 0
    distractor_hits = 0

    with torch.no_grad():
        for rec in records:
            tmpl = rec.get("template_type", "")
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]
            src_pos = rec["source_token_index"]
            cands = rec.get("candidate_token_indices", [src_pos])

            p_toks = tokenizer.encode(prompt)
            full_seq = [0] + p_toks
            seq_len = len(full_seq)
            valid_src = min(src_pos, seq_len - 1)

            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            for c in cands:
                if c < seq_len:
                    mask[0, c] = True
            mask[0, valid_src] = True

            best_pos, copied_tok_id, conf = model.point_and_copy(inp_tensor, mask)
            copied_text = tokenizer.decode([copied_tok_id]).strip()

            is_pos_hit = (best_pos == valid_src)
            is_tok_hit = (copied_tok_id == target_tok_id) or (copied_text == target_val.strip())

            if is_pos_hit:
                pos_hits += 1
            if is_tok_hit:
                token_em_hits += 1

            if tmpl == "multi_item_lookup":
                distractor_total += 1
                if is_tok_hit:
                    distractor_hits += 1

            predictions.append({
                "prompt": prompt,
                "expected": target_val,
                "generated": copied_text,
                "selected_pos": best_pos,
                "expected_pos": valid_src,
                "confidence": conf,
                "exact_match": is_tok_hit
            })

    unique_outputs = set(p["generated"] for p in predictions)
    return {
        "total": total,
        "position_acc": (pos_hits / total) * 100.0,
        "token_em": (token_em_hits / total) * 100.0,
        "distractor_total": distractor_total,
        "distractor_hits": distractor_hits,
        "distractor_em": (distractor_hits / distractor_total * 100.0) if distractor_total > 0 else 0.0,
        "output_diversity": len(unique_outputs) / total if total > 0 else 0.0,
        "sample_predictions": predictions[:5]
    }

def run_evaluation():
    print("=" * 70)
    print("EXP 09: FULL ARMS COMPARISON & EVALUATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    val_path = DATA_DIR / "val.jsonl"
    test_a_path = DATA_DIR / "test_a_familiar.jsonl"
    test_b_path = DATA_DIR / "test_b_held_out.jsonl"

    arm_a_ckpt = CHECKPOINT_DIR / "arm_a_best_checkpoint.pt"
    arm_b_ckpt = CHECKPOINT_DIR / "arm_b_best_checkpoint.pt"

    # Evaluate Baseline Parent
    print("\n[1] Evaluating Baseline Parent Checkpoint...")
    parent_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    p_st = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    parent_model.load_state_dict(p_st["model_state_dict"])
    parent_model.eval()

    parent_val = eval_causal_split(parent_model, tokenizer, val_path, device=device)
    parent_test_a = eval_causal_split(parent_model, tokenizer, test_a_path, device=device)
    parent_test_b = eval_causal_split(parent_model, tokenizer, test_b_path, device=device)

    # Evaluate Arm A
    print("\n[2] Evaluating Arm A (Broader-Vocabulary Causal SFT)...")
    arm_a_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    a_st = torch.load(arm_a_ckpt, map_location=device, weights_only=False)
    arm_a_model.load_state_dict(a_st["model_state_dict"])
    arm_a_model.eval()

    arm_a_val = eval_causal_split(arm_a_model, tokenizer, val_path, device=device)
    arm_a_test_a = eval_causal_split(arm_a_model, tokenizer, test_a_path, device=device)
    arm_a_test_b = eval_causal_split(arm_a_model, tokenizer, test_b_path, device=device)

    # Evaluate Arm B
    print("\n[3] Evaluating Arm B (Isolated Pointer Head)...")
    arm_b_model = ChakrPointerModel(ModelConfig(max_seq_len=512), freeze_backbone=False).to(device)
    b_st = torch.load(arm_b_ckpt, map_location=device, weights_only=False)
    arm_b_model.load_state_dict(b_st["model_state_dict"])
    arm_b_model.eval()

    arm_b_val = eval_pointer_split_full(arm_b_model, tokenizer, val_path, device=device)
    arm_b_test_a = eval_pointer_split_full(arm_b_model, tokenizer, test_a_path, device=device)
    arm_b_test_b = eval_pointer_split_full(arm_b_model, tokenizer, test_b_path, device=device)

    results = {
        "baseline_parent": {
            "val": parent_val,
            "test_a": parent_test_a,
            "test_b": parent_test_b
        },
        "arm_a_broad_sft": {
            "best_step": a_st.get("step"),
            "val": arm_a_val,
            "test_a": arm_a_test_a,
            "test_b": arm_a_test_b
        },
        "arm_b_pointer": {
            "best_step": b_st.get("step"),
            "val": arm_b_val,
            "test_a": arm_b_test_a,
            "test_b": arm_b_test_b
        }
    }

    with open(EXP09_DIR / "EVALUATION_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("EXP 09 EVALUATION COMPLETE — COMPARISON SUMMARY:")
    print("=" * 70)
    print(f"VAL Split (228 ex):")
    print(f"  Parent: EM {parent_val['free_gen_em']:5.1f}% | Mean Rank: {parent_val['mean_rank']:.1f}")
    print(f"  Arm A:  EM {arm_a_val['free_gen_em']:5.1f}% | Mean Rank: {arm_a_val['mean_rank']:.1f} | Distractor EM: {arm_a_val['distractor_em']:.1f}%")
    print(f"  Arm B:  EM {arm_b_val['token_em']:5.1f}% | Pos Acc:   {arm_b_val['position_acc']:.1f}% | Distractor EM: {arm_b_val['distractor_em']:.1f}%")

    print(f"\nTEST A (Familiar Tokens, Novel Contexts - 214 ex):")
    print(f"  Parent: EM {parent_test_a['free_gen_em']:5.1f}% | Mean Rank: {parent_test_a['mean_rank']:.1f}")
    print(f"  Arm A:  EM {arm_a_test_a['free_gen_em']:5.1f}% | Mean Rank: {arm_a_test_a['mean_rank']:.1f} | Distractor EM: {arm_a_test_a['distractor_em']:.1f}%")
    print(f"  Arm B:  EM {arm_b_test_a['token_em']:5.1f}% | Pos Acc:   {arm_b_test_a['position_acc']:.1f}% | Distractor EM: {arm_b_test_a['distractor_em']:.1f}%")

    print(f"\nTEST B (Held-Out Token Identities - 300 ex):")
    print(f"  Parent: EM {parent_test_b['free_gen_em']:5.1f}% | Mean Rank: {parent_test_b['mean_rank']:.1f}")
    print(f"  Arm A:  EM {arm_a_test_b['free_gen_em']:5.1f}% | Mean Rank: {arm_a_test_b['mean_rank']:.1f} | Distractor EM: {arm_a_test_b['distractor_em']:.1f}%")
    print(f"  Arm B:  EM {arm_b_test_b['token_em']:5.1f}% | Pos Acc:   {arm_b_test_b['position_acc']:.1f}% | Distractor EM: {arm_b_test_b['distractor_em']:.1f}%")

if __name__ == "__main__":
    run_evaluation()
