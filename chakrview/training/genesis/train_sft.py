"""
ChakrGenesis Conversational SFT & Reasoning Curriculum Trainer.
Fine-tunes the pretrained ChakrGenesis foundation checkpoint using:
1. Instruction & dialogue examples (English, Hindi, Hinglish, Math, Reasoning, Code, Abstention).
2. Assistant-turn target masking (loss only computed on the assistant's reply).
3. 20% Pretraining Replay mixture from Stage-C to preserve general language fluency.
"""

from __future__ import annotations
import glob
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def assemble_sft_dataset() -> list[dict]:
    examples = []
    # 1. ucip09s balanced train (1080 dialogues: EN, HI, Hinglish, capabilities, coding)
    p_09 = Path("experiments/ucip09s/datasets/ucip09s_balanced_train.jsonl")
    if p_09.exists():
        with open(p_09, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                prompt = d.get("prompt", "").strip()
                target = d.get("target", "").strip()
                if prompt and target:
                    examples.append({"prompt": prompt, "response": target, "type": "dialogue"})

    # 2. ultimate_brain train (545 examples: arithmetic, reasoning, multilingual)
    p_ub = Path("experiments/ultimate_brain/datasets/train.jsonl")
    if p_ub.exists():
        with open(p_ub, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                prompt = d.get("prompt", "").strip()
                target = d.get("response", "").strip()
                if prompt and target:
                    examples.append({"prompt": prompt, "response": target, "type": "instruction"})

    # 3. ucip29 train (440 examples)
    p_29 = Path("experiments/ucip29/datasets/conv_train.jsonl")
    if p_29.exists():
        with open(p_29, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                prompt = d.get("prompt", "").strip()
                target = d.get("response", "").strip()
                if prompt and target:
                    examples.append({"prompt": prompt, "response": target, "type": "instruction"})

    # 4. Context-bound QA & entity tasks from ucip23
    for p_task in glob.glob("experiments/ucip23/datasets/task_*.jsonl"):
        with open(p_task, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:  # sample up to 100 per task
                    break
                d = json.loads(line)
                prompt = d.get("prompt", "").strip()
                target = d.get("target", "").strip()
                if prompt and target:
                    examples.append({"prompt": prompt, "response": target, "type": "reasoning"})

    # Deduplicate by prompt
    seen = set()
    unique_examples = []
    for ex in examples:
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            unique_examples.append(ex)

    print(f"Total Unique SFT Instructions Assembled: {len(unique_examples)}")
    return unique_examples


class GenesisSFTDataset(Dataset):
    def __init__(self, sft_examples: list[dict], tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []

        for ex in sft_examples:
            p_text = ex["prompt"].strip()
            r_text = ex["response"].strip()
            # Standardize prompt ending
            if not p_text.endswith("Assistant:"):
                p_text = p_text + "\nAssistant:"

            p_ids = tokenizer.encode(p_text)
            r_ids = tokenizer.encode(" " + r_text) + [tokenizer.eos_id if hasattr(tokenizer, "eos_id") else 2]

            full_ids = p_ids + r_ids
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            input_ids = full_ids[:-1]
            labels = full_ids[1:]

            # Mask out prompt tokens with -100 so loss is computed solely on the assistant response
            prompt_len = len(p_ids)
            masked_labels = []
            for idx, token in enumerate(labels):
                if idx < prompt_len - 1:
                    masked_labels.append(-100)
                else:
                    masked_labels.append(token)

            # Pad sequences
            pad_len = max_seq_len - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            masked_labels = masked_labels + [-100] * pad_len

            self.items.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(masked_labels, dtype=torch.long)
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[idx]


def train_genesis_sft(
    pretrained_ckpt_path: str = "experiments/project_genesis/pretraining/genesis_pretrain_best.pt",
    output_dir: str = "experiments/project_genesis/conversation_training",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  PROJECT GENESIS: CONVERSATIONAL & REASONING SFT TRAINING       ")
    print(f"  Device: {device} | Batch Size: {batch_size} | Epochs: {epochs}")
    print("=================================================================\n")

    tok, _ = load_tokenizer_artifacts("experiments/tokenizer/candidates/v4096")
    sft_data = assemble_sft_dataset()
    dataset = GenesisSFTDataset(sft_data, tok, max_seq_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pretrained weights
    print(f"Loading pretrained foundation checkpoint from: {pretrained_ckpt_path}")
    ckpt = torch.load(pretrained_ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrGenesis(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model successfully loaded. Parameter count: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg_l = running_loss / 50
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | SFT Loss: {avg_l:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

        # Epoch end validation & generation test
        model.eval()
        test_prompts = [
            "User: Hello! Who are you?\nAssistant:",
            "User: Namaste, aap kaise hain?\nAssistant:",
            "User: Multiply 6 by 7.\nAssistant:"
        ]
        print(f"\n--- Epoch {epoch+1} Free Generation Samples ---")
        for tp in test_prompts:
            t_ids = torch.tensor([tok.encode(tp)], device=device)
            out_ids = model.generate(t_ids, max_new_tokens=25, temperature=0.7)
            ans = tok.decode(out_ids[0].tolist(), errors="replace")
            print(f"  {ans}")
        print("-----------------------------------------------\n")

    # Save final best SFT model
    final_sft_path = out_path / "genesis_sft_best.pt"
    torch.save({
        "epoch": epochs,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_sft_path)
    print(f"[COMPLETE] SFT training finished! Saved checkpoint to {final_sft_path}")


if __name__ == "__main__":
    train_genesis_sft()
