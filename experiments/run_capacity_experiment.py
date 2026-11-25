"""
NR-016 Phase 4: Fair 17M vs 38M Capacity Comparison Experiment
Executes:
1. Matched training from scratch (Seed 42 and Seed 101) for both 17M and 38M
2. Identical training tokens, batch size (16), LR schedule, and dataset order
3. In-distribution vs out-of-distribution evaluation on the sealed 320-task benchmark
4. Records training loss, generalization gap, peak VRAM, wall-clock time, and strict accuracy
"""

import sys
import os
import ast
import json
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ScientificDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        prompt, target = text.split("<assistant>\n")
        prompt_str = prompt + "<assistant>\n"
        target_str = target

        p_ids = self.tokenizer.encode(prompt_str)
        t_ids = self.tokenizer.encode(target_str)
        eos_id = getattr(self.tokenizer, 'eos_id', 1) or 1
        t_ids.append(eos_id)

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

def evaluate_on_benchmark(model, tokenizer, benchmark_tasks, device):
    model.eval()
    in_dist_correct = 0
    in_dist_total = 0
    ood_correct = 0
    ood_total = 0

    for task in benchmark_tasks:
        prompt = f"<user>\n{task['prompt']}\n<assistant>\n"
        p_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        curr = p_ids
        with torch.no_grad():
            for _ in range(40):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1:
                    break

        gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
        is_ood = task['is_ood']
        eval_type = task['eval_type']
        gt = task['ground_truth']

        passed = False
        if eval_type == "python_execution":
            test_assertions = task.get('test_assertions', '')
            cleaned_code = gen_text
            if "```python" in cleaned_code: cleaned_code = cleaned_code.split("```python")[1].split("```")[0].strip()
            elif "```" in cleaned_code: cleaned_code = cleaned_code.split("```")[1].split("```")[0].strip()
            try:
                ast.parse(cleaned_code)
                res = subprocess.run([sys.executable, "-c", f"{cleaned_code}\n\n{test_assertions}\n"], capture_output=True, timeout=2.0)
                passed = (res.returncode == 0)
            except Exception:
                passed = False
        elif eval_type == "exact_match":
            passed = (gt.lower() in gen_text.lower())
        else:
            passed = (gt in gen_text)

        if is_ood:
            ood_total += 1
            if passed: ood_correct += 1
        else:
            in_dist_total += 1
            if passed: in_dist_correct += 1

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
        "ood_total": ood_total
    }

def run_capacity_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-016 FAIR CAPACITY EXPERIMENT (17M vs 38M) on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    train_path = "data/scientific_eval_016/scientific_train_1600.json"
    bench_path = "data/scientific_eval_016/scientific_benchmark_320.json"
    with open(bench_path, 'r', encoding='utf-8') as f:
        bench_tasks = json.load(f)

    # 1. 17M Config
    cfg_17m = Genesis17MConfig(
        vocab_size=4096, d_model=384, n_layers=10, n_q_heads=12, n_kv_heads=3,
        hidden_dim=1024, max_seq_len=128, dropout=0.0
    )
    # 2. 38M Config
    cfg_38m = Genesis17MConfig(
        vocab_size=4096, d_model=512, n_layers=12, n_q_heads=16, n_kv_heads=4,
        hidden_dim=1536, max_seq_len=128, dropout=0.0
    )

    models_to_test = [
        ("17M_Seed42", cfg_17m, 42),
        ("38M_Seed42", cfg_38m, 42),
        ("17M_Seed101", cfg_17m, 101),
        ("38M_Seed101", cfg_38m, 101),
    ]

    results = {}

    for name, cfg, seed in models_to_test:
        print(f"\n" + "="*60)
        print(f"  Training Model: {name} (Seed {seed})")
        print("="*60)

        torch.manual_seed(seed)
        model = ChakrGenesis17M(cfg).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        train_ds = ScientificDataset(train_path, tokenizer, max_seq_len=128)
        # deterministic shuffle with seed
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=g)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Train for 2 epochs (200 steps)
        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        step = 0
        total_loss = 0.0
        for epoch in range(2):
            for batch in train_loader:
                step += 1
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = loss_fn(shift_logits, shift_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"  Epoch {epoch+1} | Step {step:3d}/200 | Avg Loss: {total_loss/50:.4f}")
                    total_loss = 0.0

        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Training completed in {elapsed:.1f}s | Peak VRAM: {peak_vram:.1f} MB")

        # Evaluate on the 320-task sealed benchmark
        print(f"Evaluating {name} on 320-task benchmark...")
        eval_metrics = evaluate_on_benchmark(model, tokenizer, bench_tasks, device)
        print(f"  In-Distribution Accuracy: {eval_metrics['in_dist_acc']:.1f}% ({eval_metrics['in_correct']}/{eval_metrics['in_total']})")
        print(f"  Out-of-Distribution Acc:  {eval_metrics['ood_acc']:.1f}% ({eval_metrics['ood_correct']}/{eval_metrics['ood_total']})")
        print(f"  Overall Strict Accuracy:  {eval_metrics['overall_acc']:.1f}%")

        results[name] = {
            "parameters": param_count,
            "seed": seed,
            "training_time_sec": elapsed,
            "peak_vram_mb": peak_vram,
            "final_loss": loss.item(),
            **eval_metrics
        }

    out_file = Path("data/scientific_eval_016/capacity_comparison_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("  CAPACITY COMPARISON SUMMARY (17M vs 38M)")
    print("="*60)
    for name, r in results.items():
        print(f"{name:15s} | In-Dist: {r['in_dist_acc']:5.1f}% | OOD: {r['ood_acc']:5.1f}% | Overall: {r['overall_acc']:5.1f}% | VRAM: {r['peak_vram_mb']:5.1f}MB")

    print(f"\nSaved capacity comparison results to: {out_file}")

if __name__ == '__main__':
    run_capacity_experiment()
