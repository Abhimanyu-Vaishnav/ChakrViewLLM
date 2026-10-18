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
from experiments.batch91_94_tool_calling_comparator.tools.calculator import validate_and_execute_tool_call

DATA_DIR = Path("experiments/batch111_114_curriculum_transfer/data")
COPY_DATA_DIR = Path("experiments/batch107_110_operand_copy_benchmark/data")
OUTPUT_DIR = Path("experiments/batch111_114_curriculum_transfer")

def load_records(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def parse_tool_call(text: str) -> dict:
    import re
    # Expect TOOL_CALL: {"tool": "...", "op": "...", "a": \d+, "b": \d+}
    m = re.search(r"TOOL_CALL:\s*(\{.*?\})", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        return data
    except Exception:
        return None

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

def evaluate_tool_model(model: ChakrMicro, tokenizer, records: list) -> dict:
    model.eval()
    total = len(records)
    valid_syntax = 0
    correct_tool = 0
    correct_op = 0
    a_correct = 0
    b_correct = 0
    pair_correct = 0
    executed_correct = 0
    generated_list = []
    detailed = []
    
    for r in records:
        a_val, b_val = r["operands"]
        prompt = r["prompt"]
        prompt_ids = [0] + tokenizer.encode(prompt)
        
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
        generated_list.append(clean_gen)
        
        parsed = parse_tool_call(clean_gen)
        is_syntax = (parsed is not None)
        is_tool = is_syntax and (parsed.get("tool") == "calculator")
        is_op = is_syntax and (parsed.get("op") == "add")
        is_a = is_syntax and (parsed.get("a") == a_val)
        is_b = is_syntax and (parsed.get("b") == b_val)
        is_pair = (is_a and is_b)
        
        is_exec = False
        if is_syntax:
            valid_syntax += 1
            if is_tool: correct_tool += 1
            if is_op: correct_op += 1
            if is_a: a_correct += 1
            if is_b: b_correct += 1
            if is_pair: pair_correct += 1
            
            # Execute through allowlisted calculator
            exec_res = validate_and_execute_tool_call(clean_gen)
            if exec_res.get("executed") and exec_res.get("result") == (a_val + b_val):
                is_exec = True
                executed_correct += 1
                
        detailed.append({
            "operands": [a_val, b_val],
            "gen": clean_gen,
            "parsed": parsed,
            "syntax_ok": is_syntax,
            "pair_ok": is_pair,
            "exec_ok": is_exec,
        })
        
    counts = collections.Counter(generated_list)
    most_common, max_count = counts.most_common(1)[0] if counts else ("", 0)
    
    return {
        "total": total,
        "syntax_rate": round(valid_syntax / max(1, total), 4),
        "tool_acc": round(correct_tool / max(1, total), 4),
        "op_acc": round(correct_op / max(1, total), 4),
        "a_acc": round(a_correct / max(1, total), 4),
        "b_acc": round(b_correct / max(1, total), 4),
        "pair_acc": round(pair_correct / max(1, total), 4),
        "exec_acc": round(executed_correct / max(1, total), 4),
        "unique_outputs": len(counts),
        "collapse_rate": round(max_count / max(1, total), 4),
        "most_common": most_common,
        "detailed": detailed,
    }

def evaluate_clarification(model: ChakrMicro, tokenizer, controls: list) -> dict:
    model.eval()
    total = len(controls)
    spurious = 0
    detailed = []
    for c in controls:
        prompt_ids = [0] + tokenizer.encode(c["prompt"])
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(50):
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
        clean = tokenizer.decode(gen_tokens[len(prompt_ids):]).split("<|im_end|>")[0].strip()
        parsed = parse_tool_call(clean)
        is_spurious = (parsed is not None)
        if is_spurious:
            spurious += 1
        detailed.append({"prompt": c["prompt"], "gen": clean, "spurious": is_spurious})
    return {
        "total": total,
        "spurious_calls": spurious,
        "spurious_rate": round(spurious / max(1, total), 4),
        "detailed": detailed,
    }

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    tool_splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    controls = load_records(DATA_DIR / "clarification_controls.jsonl")
    copy_train = load_records(COPY_DATA_DIR / "train.jsonl")
    
    # Base checkpoint weights
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_weights = base_state["model_state_dict"] if "model_state_dict" in base_state else base_state
    
    # Arm 1: Direct Tool SFT Control (Canonical Base -> Tool SFT 150 steps)
    _, _, ckpt_dir, sha_dir = train_arm(
        tool_splits["train"], tok, "direct_tool_control", base_weights, steps=150, batch_size=4, lr=3e-4, seed=42
    )
    
    # Arm 2: Curriculum Bridge SFT (Copy Task Checkpoint -> Tool SFT 150 steps)
    copy_ckpt_path = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")
    copy_state = torch.load(copy_ckpt_path, map_location="cpu", weights_only=False)
    copy_weights = copy_state["model_state_dict"]
    
    _, _, ckpt_curr, sha_curr = train_arm(
        tool_splits["train"], tok, "curriculum_bridge_transfer", copy_weights, steps=150, batch_size=4, lr=3e-4, seed=42
    )
    
    # Cold reload evaluations for both arms
    cfg = ModelConfig(max_seq_len=512)
    arms = [
        ("direct_tool_control", ckpt_dir, sha_dir),
        ("curriculum_bridge_transfer", ckpt_curr, sha_curr),
    ]
    
    results = {}
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
            "clarification": evaluate_clarification(m, tok, controls),
        }
        for s_name in ["train", "val", "test"]:
            eval_dict = evaluate_tool_model(m, tok, tool_splits[s_name])
            arm_eval["splits"][s_name] = eval_dict
            print(f"[{s_name:5s}] Syntax={eval_dict['syntax_rate']*100:.1f}%, PairAcc={eval_dict['pair_acc']*100:.1f}%, ExecAcc={eval_dict['exec_acc']*100:.1f}%, Collapse={eval_dict['collapse_rate']*100:.1f}%")
            print(f"        MostCommon: {repr(eval_dict['most_common'][:60])}")
        results[arm_name] = arm_eval
        
    out_file = OUTPUT_DIR / "reports" / "curriculum_transfer_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Curriculum transfer results saved to {out_file}")

if __name__ == "__main__":
    main()
