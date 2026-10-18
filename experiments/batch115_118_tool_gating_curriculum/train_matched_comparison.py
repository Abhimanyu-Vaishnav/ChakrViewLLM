import json
import collections
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    evaluate_single_prediction,
    compute_aggregate_metrics,
)

DATA_DIR = Path("experiments/batch115_118_tool_gating_curriculum/data")
OUTPUT_DIR = Path("experiments/batch115_118_tool_gating_curriculum")

def load_records(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_training_sequence(record: dict, tokenizer):
    prompt_tokens = tokenizer.encode(record["prompt"])
    full_tokens = tokenizer.encode(record["full_conversation"])
    input_ids = [0] + full_tokens + [1]
    target_ids = [-100] * (len(prompt_tokens) + 1) + full_tokens[len(prompt_tokens):] + [1]
    return input_ids, target_ids

def train_arm(
    train_records: list,
    tokenizer,
    condition_name: str,
    base_state_dict: dict,
    steps: int = 150,
    batch_size: int = 4,
    lr: float = 3e-4,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Training: {condition_name} ({steps} steps)")
    print(f"{'='*60}")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    
    prepared = [build_training_sequence(r, tokenizer) for r in train_records]
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(base_state_dict)
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    history = []
    model.train()
    for step in range(steps):
        batch_indices = rng.randint(0, len(prepared), size=batch_size)
        batch = [prepared[i] for i in batch_indices]
        
        max_len = max(len(inp) for inp, _ in batch)
        x_pad = torch.full((batch_size, max_len), 0, dtype=torch.long)
        y_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for b, (inp, tgt) in enumerate(batch):
            x_pad[b, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            y_pad[b, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            
        x_in = x_pad[:, :-1]
        y_tgt = y_pad[:, 1:]
        
        opt.zero_grad()
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    ckpt_path = OUTPUT_DIR / "checkpoints" / f"{condition_name}_step{steps}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": condition_name,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved: {ckpt_path} (SHA-256: {sha})")
    return model, history, ckpt_path, sha

def evaluate_on_records(model: ChakrMicro, tokenizer, records: list) -> dict:
    model.eval()
    eval_list = []
    
    for r in records:
        prompt_ids = [0] + tokenizer.encode(r["prompt"])
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(50):
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
        raw_gen = tokenizer.decode(gen_tokens[len(prompt_ids):])
        clean_gen = raw_gen.split("<|im_end|>")[0].strip()
        
        res = evaluate_single_prediction(clean_gen, r)
        res["id"] = r["id"]
        res["prompt_snippet"] = r["prompt"][:60]
        eval_list.append(res)
        
    agg = compute_aggregate_metrics(eval_list)
    agg["per_example_details"] = eval_list
    return agg

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    
    # 1. Base weights from canonical checkpoint
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_weights = base_state["model_state_dict"] if "model_state_dict" in base_state else base_state
    
    # 2. Copy task weights from verified Stage 1 copy checkpoint
    copy_ckpt_path = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")
    copy_state = torch.load(copy_ckpt_path, map_location="cpu", weights_only=False)
    copy_weights = copy_state["model_state_dict"]
    
    # Train Arm 1: Control (Canonical Base -> Balanced Curriculum 150 steps)
    _, _, ckpt_ctrl, sha_ctrl = train_arm(
        splits["train"], tok, "balanced_control_base", base_weights, steps=150, batch_size=4, lr=3e-4, seed=42
    )
    
    # Train Arm 2: Curriculum (Stage 1 Copy -> Balanced Curriculum 150 steps)
    _, _, ckpt_curr, sha_curr = train_arm(
        splits["train"], tok, "balanced_curriculum_bridge", copy_weights, steps=150, batch_size=4, lr=3e-4, seed=42
    )
    
    # Cold reload evaluation
    cfg = ModelConfig(max_seq_len=512)
    arms = [
        ("balanced_control_base", ckpt_ctrl, sha_ctrl),
        ("balanced_curriculum_bridge", ckpt_curr, sha_curr),
    ]
    
    final_report = {}
    for arm_name, ckpt_p, sha in arms:
        print(f"\n--- Cold Reload Evaluation: {arm_name} ---")
        m = ChakrMicro(cfg)
        st = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        m.load_state_dict(st["model_state_dict"])
        m.eval()
        
        arm_eval = {
            "checkpoint": str(ckpt_p),
            "sha256": sha,
            "splits": {},
        }
        for s_name in ["train", "val", "test"]:
            res = evaluate_on_records(m, tok, splits[s_name])
            arm_eval["splits"][s_name] = res
            tm = res.get("tool_metrics", {})
            ntm = res.get("non_tool_metrics", {})
            print(f"[{s_name:5s}] Tool n={res['tool_examples_count']}: Syntax={tm.get('parse_rate')}, PairAcc={tm.get('exact_call_accuracy')}, ExecAcc={tm.get('execution_success_rate')}, E2E={tm.get('e2e_correctness_rate')}, Collapse={tm.get('collapse_rate')}")
            print(f"        NonTool n={res['non_tool_examples_count']}: FalsePositive={ntm.get('false_positive_tool_call_rate')}, CleanRefusal={ntm.get('clean_refusal_rate')}")
            
        final_report[arm_name] = arm_eval
        
    out_file = OUTPUT_DIR / "reports" / "matched_comparison_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[OK] Matched comparison results saved to {out_file}")

if __name__ == "__main__":
    main()
