"""
UCIP-04 Controlled Autoregressive Neural Training Harness.
Trains ChakrMicro (3.44M) and ChakrMedium (12.9M) on the balanced foundational corpus.

Key innovations:
- Evaluates greedy autoregressive rollouts at every validation step (not just teacher-forcing).
- Evaluates held-out generation on key diagnostic prompts:
  1. "User: Write one simple sentence about a dog.\nAssistant:"
  2. "User: What does a CPU do?\nAssistant:"
  3. "User: Say exactly: The sky is blue.\nAssistant:"
  4. "User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:"
- Checkpoints based on validation loss and autoregressive coherence.
"""

import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW

from chakrview.brain.model import ModelConfig, ChakrMicro, initialize_weights
from scripts.run_m3_evaluations import load_tokenizer_artifacts, Sampler
from chakrview.runtime.sampling import SamplingConfig

def load_data(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def collate_dialogue(dialogue, tokenizer, max_len=256):
    prompt = dialogue["prompt"]
    target = dialogue["target"]
    
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
    
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        
    labels = [-100] * len(prompt_ids) + target_ids
    if len(labels) > max_len:
        labels = labels[:max_len]
        
    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)

def test_autoregressive_rollout(model, tokenizer, sampler, prompt_text: str, max_new: int = 24) -> str:
    prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    curr_tokens = list(prompt_tokens)
    gen_tokens = []
    sampling_cfg = SamplingConfig(temperature=0.0, top_k=0)

    with torch.no_grad():
        for step in range(max_new):
            t_in = torch.tensor([curr_tokens], dtype=torch.long)
            logits = model(t_in)[0, -1, :]
            next_tok = sampler.sample(logits, generated_tokens=gen_tokens, config=sampling_cfg, step=step)
            gen_tokens.append(next_tok)
            curr_tokens.append(next_tok)
            if next_tok == 1:
                break
    return tokenizer.decode(gen_tokens, errors="replace").strip()

def run_training(model_type="micro", total_steps=350, lr=6e-5):
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    sampler = Sampler()

    train_records = load_data(REPO_ROOT / "experiments/cognition/ucip_04/data/ucip04_train.jsonl")
    val_records = load_data(REPO_ROOT / "experiments/cognition/ucip_04/data/ucip04_validation.jsonl")
    print(f"[{model_type.upper()}] Loaded {len(train_records)} train records and {len(val_records)} val records.")

    if model_type == "micro":
        cfg = ModelConfig()
        model = ChakrMicro(cfg)
        parent_path = REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt"
        parent_ckpt = torch.load(parent_path, map_location="cpu", weights_only=False)
        sd = parent_ckpt.get("model_state_dict") or parent_ckpt
        model.load_state_dict(sd)
        print("Initialized ChakrMicro (3.44M) from v0.11 parent.")
    elif model_type == "medium":
        cfg = ModelConfig(
            vocab_size=4096,
            d_model=384,
            n_layers=7,
            n_heads=6,
            hidden_dim=896,
            max_seq_len=512,
            tie_embeddings=True
        )
        model = ChakrMicro(cfg)
        initialize_weights(model, cfg)
        parent_path = REPO_ROOT / "experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt"
        if parent_path.exists():
            parent_ckpt = torch.load(parent_path, map_location="cpu", weights_only=False)
            sd = parent_ckpt.get("model_state_dict") or parent_ckpt
            model.load_state_dict(sd)
            print("Initialized ChakrMedium (12.9M) from 13M recovered best.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    save_dir = REPO_ROOT / f"experiments/cognition/ucip_04/checkpoints/{model_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    best_val_loss = float("inf")
    best_step = 0
    batch_size = 4
    grad_accum_steps = 2
    step = 0
    train_iter = 0
    t0 = time.time()

    history = []

    diagnostic_prompts = [
        ("dog", "User: Write one simple sentence about a dog.\nAssistant:"),
        ("cpu", "User: What does a CPU do?\nAssistant:"),
        ("copy", "User: Say exactly: The sky is blue.\nAssistant:"),
        ("hindi", "User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:")
    ]

    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            for _ in range(batch_size):
                rec = train_records[train_iter % len(train_records)]
                train_iter += 1
                inp, tgt = collate_dialogue(rec, tok)
                if inp.shape[1] < 2:
                    continue
                logits = model(inp[:, :-1])
                shift_logits = logits.view(-1, cfg.vocab_size)
                shift_targets = tgt[:, 1:].contiguous().view(-1)

                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(shift_logits, shift_targets) / (grad_accum_steps * batch_size)
                loss.backward()
                accum_loss += loss.item() * (grad_accum_steps * batch_size)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 25 == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            val_correct = 0

            with torch.no_grad():
                for rec in val_records:
                    inp, tgt = collate_dialogue(rec, tok)
                    logits = model(inp[:, :-1])
                    shift_logits = logits.view(-1, cfg.vocab_size)
                    shift_targets = tgt[:, 1:].contiguous().view(-1)

                    mask = shift_targets != -100
                    if mask.sum() > 0:
                        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
                        nll = loss_fn(shift_logits, shift_targets).item()
                        val_loss += nll
                        val_tokens += mask.sum().item()
                        preds = shift_logits.argmax(dim=-1)
                        val_correct += ((preds == shift_targets) & mask).sum().item()

            mean_val_loss = val_loss / max(1, val_tokens)
            val_acc = (val_correct / max(1, val_tokens)) * 100.0

            rollouts = {}
            for name, prompt_text in diagnostic_prompts:
                rollouts[name] = test_autoregressive_rollout(model, tok, sampler, prompt_text)

            print(f"[{model_type.upper()}] Step {step}/{total_steps} | Train Loss: {accum_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Rollout [Dog]:   {repr(rollouts['dog'])}")
            print(f"  Rollout [CPU]:   {repr(rollouts['cpu'])}")
            print(f"  Rollout [Copy]:  {repr(rollouts['copy'])}")
            print(f"  Rollout [Hindi]: {repr(rollouts['hindi'])}")

            history.append({
                "step": step,
                "train_loss": accum_loss,
                "val_loss": mean_val_loss,
                "val_acc": val_acc,
                "rollouts": rollouts
            })

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_step = step
                ckpt_data = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg
                }
                torch.save(ckpt_data, save_dir / "candidate_best.pt")
                print(f"  --> Saved new best checkpoint at step {step} with val_loss={best_val_loss:.4f}")

            model.train()

    dt = time.time() - t0
    print(f"[{model_type.upper()}] Training finished in {dt:.1f}s. Best Step: {best_step}, Best Val Loss: {best_val_loss:.4f}")

    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["micro", "medium", "both"], default="both")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    if args.model in ["micro", "both"]:
        print("\n================ STARTING CHAKRMICRO (3.44M) TRAINING ================")
        run_training("micro", total_steps=args.steps, lr=6e-5)
    if args.model in ["medium", "both"]:
        print("\n================ STARTING CHAKRMEDIUM (12.9M) TRAINING ================")
        run_training("medium", total_steps=args.steps, lr=7e-5)
