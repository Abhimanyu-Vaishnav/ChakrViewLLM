import json
import numpy as np
import torch
from pathlib import Path

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    ChakrMicro, ModelConfig, load_tokenizer_artifacts, TOKENIZER_DIR, A_VALUE_TOKENS, B_VALUE_TOKENS
)
from experiments.batch95_98_slot_binding_diagnostics.experiments.run_slot_binding_comparison import load_records, DATA_DIR

tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
cfg = ModelConfig()

ckpts = {
    "Condition A (Condition C baseline)": "experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt",
    "Condition B (Uniform-loss control)": "experiments/batch103_106_slot_weighted_sft/checkpoints/uniform_loss_control_stream_step100.pt",
    "Condition C (Slot-weighted 5x)": "experiments/batch103_106_slot_weighted_sft/checkpoints/slot_weighted_5x_stream_step100.pt",
}

test_recs = load_records(DATA_DIR / "test.jsonl")

summary_report = {}

for label, ckpt_path in ckpts.items():
    model = ChakrMicro(cfg)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    a_probs, b_probs = [], []
    a_ranks, b_ranks = [], []
    a_top_tokens, b_top_tokens = [], []

    for r in test_recs:
        a_val, b_val = r["operands"]
        target_a_tok = A_VALUE_TOKENS[a_val][0]
        target_b_tok = B_VALUE_TOKENS[b_val][0]

        # Teacher-forced prefix up to operand a value
        prefix_a = r["prompt"] + 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": '
        toks_a = [0] + tok.encode(prefix_a)
        with torch.no_grad():
            logits_a = model(torch.tensor([toks_a]))[0, -1]
            probs_a = torch.softmax(logits_a, dim=-1)
            rank_a = (logits_a > logits_a[target_a_tok]).sum().item() + 1
            a_probs.append(probs_a[target_a_tok].item())
            a_ranks.append(rank_a)
            a_top_tokens.append(torch.argmax(logits_a).item())

        prefix_b = prefix_a + str(a_val) + ', "b": '
        toks_b = [0] + tok.encode(prefix_b)
        with torch.no_grad():
            logits_b = model(torch.tensor([toks_b]))[0, -1]
            probs_b = torch.softmax(logits_b, dim=-1)
            rank_b = (logits_b > logits_b[target_b_tok]).sum().item() + 1
            b_probs.append(probs_b[target_b_tok].item())
            b_ranks.append(rank_b)
            b_top_tokens.append(torch.argmax(logits_b).item())

    modal_a_tok = int(np.bincount(a_top_tokens).argmax())
    modal_b_tok = int(np.bincount(b_top_tokens).argmax())
    report_entry = {
        "operand_a": {
            "mean_target_prob": round(float(np.mean(a_probs)), 4),
            "std_target_prob": round(float(np.std(a_probs)), 4),
            "mean_rank": round(float(np.mean(a_ranks)), 2),
            "median_rank": float(np.median(a_ranks)),
            "modal_top_token_id": modal_a_tok,
            "modal_top_token_str": tok.decode([modal_a_tok]),
        },
        "operand_b": {
            "mean_target_prob": round(float(np.mean(b_probs)), 4),
            "std_target_prob": round(float(np.std(b_probs)), 4),
            "mean_rank": round(float(np.mean(b_ranks)), 2),
            "median_rank": float(np.median(b_ranks)),
            "modal_top_token_id": modal_b_tok,
            "modal_top_token_str": tok.decode([modal_b_tok]),
        }
    }
    summary_report[label] = report_entry
    print(f"=== {label} ===")
    print(f"  Operand A: Mean target prob = {np.mean(a_probs)*100:.2f}%, Mean rank = {np.mean(a_ranks):.1f}, Top token modal = {modal_a_tok} ({repr(tok.decode([modal_a_tok]))})")
    print(f"  Operand B: Mean target prob = {np.mean(b_probs)*100:.2f}%, Mean rank = {np.mean(b_ranks):.1f}, Top token modal = {modal_b_tok} ({repr(tok.decode([modal_b_tok]))})")

out_file = Path("experiments/batch103_106_slot_weighted_sft/reports/teacher_forced_eval.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=2)
print(f"\nSaved teacher forced evaluation to {out_file}")
