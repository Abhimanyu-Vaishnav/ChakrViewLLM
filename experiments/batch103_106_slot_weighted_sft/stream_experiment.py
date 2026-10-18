import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch95_98_slot_binding_diagnostics.experiments.run_slot_binding_comparison import (
    evaluate_slot_binding_on_records, evaluate_clarification_controls, load_records, DATA_DIR
)

tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
with open("experiments/batch95_98_slot_binding_diagnostics/data/train_slot_tokens.bin", "rb") as f:
    tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int64)

# Create slot mask over all tokens in the stream
# Targets at slot positions are the values of 'a' and 'b' in TOOL_CALL
TOK_QUOTE, TOK_A, TOK_B, TOK_COLON_SP = 37, 100, 101, 544
slot_mask = np.zeros(len(tokens), dtype=bool)
for i in range(len(tokens) - 5):
    if tokens[i] == TOK_QUOTE and tokens[i+2] == TOK_COLON_SP and tokens[i+1] in (TOK_A, TOK_B):
        j = i + 3
        while j < len(tokens) and tokens[j] != TOK_QUOTE and j < i + 6:
            slot_mask[j] = True
            j += 1

print(f"Total tokens: {len(tokens)}, Total slot tokens marked: {np.sum(slot_mask)} ({100*np.sum(slot_mask)/len(tokens):.2f}%)")

splits = {
    "train": load_records(DATA_DIR / "train.jsonl"),
    "val": load_records(DATA_DIR / "val.jsonl"),
    "test": load_records(DATA_DIR / "test.jsonl"),
}
controls = load_records(DATA_DIR / "clarification_controls.jsonl")

results = {}

for weight_slots in [False, True]:
    cond_name = "slot_weighted_5x" if weight_slots else "uniform_loss_control"
    print(f"\n============================================================")
    print(f"Training: {cond_name} (weight_slots={weight_slots})")
    print(f"============================================================")
    t0 = time.time()
    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"])
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    steps = 100
    batch_size = 4
    seq_len = 64
    max_idx = len(tokens) - seq_len - 1

    model.train()
    history = []
    slot_steps_count = 0
    for step in range(steps):
        idxs = rng.randint(0, max_idx, size=batch_size)
        x = torch.stack([torch.tensor(tokens[i : i + seq_len], dtype=torch.long) for i in idxs])
        y = torch.stack([torch.tensor(tokens[i + 1 : i + seq_len + 1], dtype=torch.long) for i in idxs])

        opt.zero_grad()
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, y.unsqueeze(2)).squeeze(2)

        if weight_slots:
            w_mat = np.ones((batch_size, seq_len), dtype=np.float32)
            for b, i in enumerate(idxs):
                target_indices = np.arange(i + 1, i + seq_len + 1)
                is_slot = slot_mask[target_indices]
                w_mat[b, is_slot] = 5.0
                if np.any(is_slot):
                    slot_steps_count += 1
            weights = torch.tensor(w_mat, dtype=torch.float32)
            loss = -(weights * target_log_probs).sum() / weights.sum()
        else:
            loss = -target_log_probs.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})

    ckpt_path = Path("experiments/batch103_106_slot_weighted_sft/checkpoints") / f"{cond_name}_stream_step100.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": cond_name,
        "weighted": weight_slots,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved: {ckpt_path} (SHA-256: {sha})")

    # Evaluate
    model.eval()
    cond_results = {
        "checkpoint": str(ckpt_path),
        "sha256": sha,
        "splits": {},
        "clarification_control": evaluate_clarification_controls(model, tok, controls),
    }
    for split_name in ["train", "val", "test"]:
        eval_dict = evaluate_slot_binding_on_records(model, tok, splits[split_name])
        # delete full traces from summary dict for brevity in logs
        traces = eval_dict.pop("traces", [])
        cond_results["splits"][split_name] = eval_dict
        print(f"[{split_name:5s}] Syntax: {eval_dict['valid_tool_syntax_rate']*100:.1f}%, ArgAcc: {eval_dict['exact_argument_pair_rate']*100:.1f}%, Collapse: {eval_dict['constant_call_collapse_rate']*100:.1f}%, MostCommon: {eval_dict['most_common_call']}")

    results[cond_name] = cond_results

# Also evaluate Condition C for reference (Condition A)
cond_c_ckpt = Path("experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt")
if cond_c_ckpt.exists():
    print("\n--- Evaluating Condition C (Condition A benchmark) ---")
    model_c = ChakrMicro(cfg)
    c_state = torch.load(cond_c_ckpt, map_location="cpu", weights_only=False)
    model_c.load_state_dict(c_state["model_state_dict"])
    model_c.eval()
    c_results = {
        "checkpoint": str(cond_c_ckpt),
        "sha256": compute_file_sha256(str(cond_c_ckpt)),
        "splits": {},
        "clarification_control": evaluate_clarification_controls(model_c, tok, controls),
    }
    for split_name in ["train", "val", "test"]:
        eval_dict = evaluate_slot_binding_on_records(model_c, tok, splits[split_name])
        eval_dict.pop("traces", None)
        c_results["splits"][split_name] = eval_dict
        print(f"[{split_name:5s}] Syntax: {eval_dict['valid_tool_syntax_rate']*100:.1f}%, ArgAcc: {eval_dict['exact_argument_pair_rate']*100:.1f}%, Collapse: {eval_dict['constant_call_collapse_rate']*100:.1f}%, MostCommon: {eval_dict['most_common_call']}")
    results["condition_c_baseline"] = c_results

out_report = Path("experiments/batch103_106_slot_weighted_sft/reports/controlled_experiment_results.json")
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[OK] Controlled results saved to {out_report}")
