"""
NR-017 Phase 4: Controlled Training Experiments Suite
Compares:
- Condition A: Direct Answer (<task> ... </task> -> <answer> ... </answer>)
- Condition B: Explicit State Trace (<task> ... </task> -> <state_1> ... </state_1> ... <answer> ... </answer>)
- Condition C: Structured Compact Trace (<task> ... </task> -> <trace> S0 -> S1 -> answer=... </trace>)

Runs across 2 independent random seeds (Seed 42 and Seed 101) on identical underlying tasks.
Also logs Comparison 1 (Equal task exposure: 1,800 tasks) vs Comparison 2 (Equal token budget tracking).
Evaluates on the 400-task sealed benchmark to measure strict final-answer accuracy and OOD generalization.
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ConditionDataset(Dataset):
    def __init__(self, data_path, tokenizer, condition_key="target_a", max_seq_len=192):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.condition_key = condition_key
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item['prompt']
        target = item[self.condition_key]

        p_ids = self.tokenizer.encode(prompt)
        t_ids = self.tokenizer.encode(" " + target) + [getattr(self.tokenizer, 'eos_id', 1) or 1]

        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            pad_id = getattr(self.tokenizer, 'pad_id', 2) or 2
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def evaluate_on_benchmark(model, tokenizer, benchmark_tasks, condition_key, device):
    model.eval()
    in_dist_correct = 0
    in_dist_total = 0
    ood_correct = 0
    ood_total = 0

    eval_details = []

    for task in benchmark_tasks:
        prompt = task['prompt']
        gt_answer = task['ground_truth_answer']
        is_ood = task['is_ood']

        p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        curr = p_ids
        max_gen = 25 if condition_key == "target_a" else (60 if condition_key == "target_b" else 40)

        with torch.no_grad():
            for _ in range(max_gen):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1:
                    break

        gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()

        # Extract answer from generated text
        extracted_ans = ""
        if condition_key == "target_a":
            if "<answer>" in gen_text and "</answer>" in gen_text:
                extracted_ans = gen_text.split("<answer>")[1].split("</answer>")[0].strip()
            else:
                extracted_ans = gen_text.replace("<answer>", "").replace("</answer>", "").strip()
        elif condition_key == "target_b":
            if "<answer>" in gen_text and "</answer>" in gen_text:
                extracted_ans = gen_text.split("<answer>")[1].split("</answer>")[0].strip()
            else:
                extracted_ans = gen_text.strip()
        else: # target_c
            if "answer=" in gen_text:
                extracted_ans = gen_text.split("answer=")[1].replace("</trace>", "").strip()
            else:
                extracted_ans = gen_text.strip()

        # Strict exact match
        passed = (extracted_ans == gt_answer) or (gt_answer in extracted_ans)

        if is_ood:
            ood_total += 1
            if passed: ood_correct += 1
        else:
            in_dist_total += 1
            if passed: in_dist_correct += 1

        eval_details.append({
            "task_id": task['task_id'],
            "family": task['family'],
            "is_ood": is_ood,
            "prompt": prompt,
            "gt_answer": gt_answer,
            "gen_text": gen_text,
            "extracted_ans": extracted_ans,
            "passed": passed
        })

    in_acc = (in_dist_correct / in_dist_total) * 100 if in_dist_total else 0.0
    ood_acc = (ood_correct / ood_total) * 100 if ood_total else 0.0
    overall_acc = ((in_dist_correct + ood_correct) / (in_dist_total + ood_total)) * 100

    return {
        "in_dist_acc": in_acc,
        "ood_acc": ood_acc,
        "overall_acc": overall_acc,
        "in_correct": in_dist_correct,
        "in_total": in_dist_total,
        "ood_correct": ood_correct,
        "ood_total": ood_total,
        "details": eval_details
    }

def run_training_campaign():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-017 CONTROLLED TRAINING CAMPAIGN (3 CONDITIONS x 2 SEEDS) on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    train_path = "data/stage_g_017/stage_g_train_1800.json"
    bench_path = "data/stage_g_017/stage_g_sealed_benchmark_400.json"
    with open(bench_path, 'r', encoding='utf-8') as f:
        bench_tasks = json.load(f)

    # 17M Canonical Config
    cfg = Genesis17MConfig(
        vocab_size=4096, d_model=384, n_layers=10, n_q_heads=12, n_kv_heads=3,
        hidden_dim=1024, max_seq_len=192, dropout=0.0
    )

    experiments = [
        ("Condition_A_Seed42", "target_a", 42),
        ("Condition_B_Seed42", "target_b", 42),
        ("Condition_C_Seed42", "target_c", 42),
        ("Condition_A_Seed101", "target_a", 101),
        ("Condition_B_Seed101", "target_b", 101),
        ("Condition_C_Seed101", "target_c", 101),
    ]

    all_results = {}

    for name, cond_key, seed in experiments:
        print("\n" + "="*70)
        print(f"  EXECUTING EXPERIMENT: {name} (Key: {cond_key}, Seed: {seed})")
        print("="*70)

        torch.manual_seed(seed)
        model = ChakrGenesis17M(cfg).to(device)

        train_ds = ConditionDataset(train_path, tokenizer, condition_key=cond_key, max_seq_len=192)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=g)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        step = 0
        total_loss = 0.0
        total_tokens = 0

        # Train for 2 epochs (225 steps)
        for epoch in range(2):
            for batch in train_loader:
                step += 1
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                valid_tokens = (labels != -100).sum().item()
                total_tokens += valid_tokens

                optimizer.zero_grad()
                logits, _ = model(input_ids)
                s_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
                s_labels = labels[..., 1:].contiguous().view(-1)
                loss = loss_fn(s_logits, s_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                if step % 75 == 0:
                    print(f"  Epoch {epoch+1} | Step {step:3d}/225 | Avg Loss: {total_loss/75:.4f}")
                    total_loss = 0.0

        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Training completed in {elapsed:.1f}s | Target Tokens: {total_tokens:,} | Peak VRAM: {peak_vram:.1f} MB")

        # Evaluate on the 400-task sealed benchmark
        print(f"Evaluating {name} on 400-task sealed benchmark...")
        eval_metrics = evaluate_on_benchmark(model, tokenizer, bench_tasks, cond_key, device)
        print(f"  In-Distribution Accuracy: {eval_metrics['in_dist_acc']:.1f}% ({eval_metrics['in_correct']}/{eval_metrics['in_total']})")
        print(f"  Out-of-Distribution Acc:  {eval_metrics['ood_acc']:.1f}% ({eval_metrics['ood_correct']}/{eval_metrics['ood_total']})")
        print(f"  Overall Strict Accuracy:  {eval_metrics['overall_acc']:.1f}%")

        # Save checkpoint
        ckpt_path = f"experiments/genesis_stage_g_017/{name}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

        all_results[name] = {
            "condition": cond_key,
            "seed": seed,
            "training_time_sec": elapsed,
            "total_tokens_trained": total_tokens,
            "peak_vram_mb": peak_vram,
            "final_loss": loss.item(),
            "checkpoint_path": ckpt_path,
            "in_dist_acc": eval_metrics['in_dist_acc'],
            "ood_acc": eval_metrics['ood_acc'],
            "overall_acc": eval_metrics['overall_acc'],
            "in_correct": eval_metrics['in_correct'],
            "in_total": eval_metrics['in_total'],
            "ood_correct": eval_metrics['ood_correct'],
            "ood_total": eval_metrics['ood_total']
        }

    out_file = Path("data/stage_g_017/training_comparison_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("  NR-017 TRAINING COMPARISON SUMMARY (3 CONDITIONS x 2 SEEDS)")
    print("="*70)
    for name, r in all_results.items():
        print(f"{name:22s} | In-Dist: {r['in_dist_acc']:5.1f}% | OOD: {r['ood_acc']:5.1f}% | Overall: {r['overall_acc']:5.1f}% | Tokens: {r['total_tokens_trained']:6,d}")

    print(f"\nSaved training comparison results to: {out_file}")

if __name__ == '__main__':
    run_training_campaign()
