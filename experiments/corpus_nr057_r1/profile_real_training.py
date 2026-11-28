"""
Multi-Component Training Throughput and Latency Profiling for NR-057-R1.
Separates:
1. Forward-only throughput
2. Forward + Backward throughput
3. Full Optimizer-Step throughput (Forward + Backward + AdamW step + zero_grad)
4. Dataloader overhead throughput
5. Checkpoint serialization latency & size
6. Peak GPU VRAM and System RAM utilization
Profiles both ChakrGenesis17M and ChakrGenesis38M on CUDA.
"""

import os
import sys
import time
import json
import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig

class DummyTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
        self.targets = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def run_multi_component_profile():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA accelerator not found")
        
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    print(f"[Profiling] GPU: {gpu_name} ({gpu_total_vram_gb:.2f} GB total VRAM)")
    process = psutil.Process()

    models_to_profile = [
        ("ChakrGenesis17M", ChakrGenesis17M, Genesis17MConfig(), 16, 512),
        ("ChakrGenesis38M", ChakrGenesis38M, Genesis38MConfig(), 8, 512),
        ("ChakrGenesis38M", ChakrGenesis38M, Genesis38MConfig(), 16, 512),
    ]

    results = {
        "mission": "NR-057-R1",
        "gpu_name": gpu_name,
        "total_vram_gb": round(gpu_total_vram_gb, 2),
        "profiles": []
    }

    steps = 40
    warmup_steps = 5

    for name, model_cls, cfg, batch_size, seq_len in models_to_profile:
        print(f"\nProfiling {name} (batch={batch_size}, seq_len={seq_len})...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = model_cls(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        vocab_size = cfg.vocab_size

        ds = DummyTokenDataset(vocab_size, seq_len, num_samples=batch_size * (steps + warmup_steps + 10))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        loader_iter = iter(loader)

        # 1. Measure DataLoader Throughput alone
        t0 = time.perf_counter()
        for _ in range(steps):
            x, y = next(loader_iter)
        dl_elapsed = time.perf_counter() - t0
        dl_tokens_per_sec = (batch_size * seq_len * steps) / dl_elapsed

        # Reset iterator
        loader_iter = iter(loader)

        # Warmup
        for _ in range(warmup_steps):
            x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

        # 2. Forward-Only Throughput
        torch.cuda.synchronize(device)
        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                x, y = next(loader_iter)
                x = x.to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(x)
        torch.cuda.synchronize(device)
        fwd_elapsed = time.perf_counter() - t_fwd_start
        fwd_tok_per_sec = (batch_size * seq_len * steps) / fwd_elapsed

        # 3. Forward + Backward Throughput
        loader_iter = iter(loader)
        torch.cuda.synchronize(device)
        t_fwd_bwd_start = time.perf_counter()
        for _ in range(steps):
            x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
        torch.cuda.synchronize(device)
        fwd_bwd_elapsed = time.perf_counter() - t_fwd_bwd_start
        fwd_bwd_tok_per_sec = (batch_size * seq_len * steps) / fwd_bwd_elapsed

        # 4. Full Optimizer Step (Forward + Backward + Optimizer Step)
        loader_iter = iter(loader)
        torch.cuda.synchronize(device)
        t_opt_start = time.perf_counter()
        for _ in range(steps):
            x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        opt_elapsed = time.perf_counter() - t_opt_start
        full_step_tok_per_sec = (batch_size * seq_len * steps) / opt_elapsed
        full_step_latency_ms = (opt_elapsed / steps) * 1000

        # Memory and Checkpoints
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        sys_ram_gb = process.memory_info().rss / (1024 ** 3)

        temp_pt = f"experiments/corpus_nr057_r1/temp_chkpt_{name}.pt"
        t_save_start = time.perf_counter()
        torch.save(model.state_dict(), temp_pt)
        save_ms = (time.perf_counter() - t_save_start) * 1000
        size_mb = os.path.getsize(temp_pt) / (1024 * 1024)
        if os.path.exists(temp_pt):
            os.remove(temp_pt)

        # Projections over Corpus V6 Train (4,163,555 tokens)
        v6_train_tokens = 4163555
        epoch_seconds = v6_train_tokens / full_step_tok_per_sec
        epoch_minutes = epoch_seconds / 60
        hours_50m = (50000000 / full_step_tok_per_sec) / 3600

        del model, optimizer
        torch.cuda.empty_cache()

        item = {
            "model_name": name,
            "parameters": sum(p.numel() for p in model_cls(cfg).parameters()),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "forward_only_tokens_per_sec": round(fwd_tok_per_sec, 1),
            "forward_backward_tokens_per_sec": round(fwd_bwd_tok_per_sec, 1),
            "full_optimizer_step_tokens_per_sec": round(full_step_tok_per_sec, 1),
            "step_latency_ms": round(full_step_latency_ms, 2),
            "dataloader_throughput_tokens_per_sec": round(dl_tokens_per_sec, 1),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "vram_utilization_pct": round(peak_vram_gb / gpu_total_vram_gb * 100, 2),
            "system_ram_gb": round(sys_ram_gb, 2),
            "checkpoint_save_ms": round(save_ms, 2),
            "checkpoint_size_mb": round(size_mb, 2),
            "time_for_1_epoch_corpus_v6_minutes": round(epoch_minutes, 2),
            "time_for_50m_tokens_hours": round(hours_50m, 2)
        }
        results["profiles"].append(item)
        print(f"Result for {name} (B={batch_size}): Full Step = {item['full_optimizer_step_tokens_per_sec']} tok/s, VRAM = {item['peak_vram_gb']} GB")

    out_file = "experiments/corpus_nr057_r1/training_profile.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMulti-component profiling complete and written to {out_file}")

if __name__ == "__main__":
    run_multi_component_profile()
