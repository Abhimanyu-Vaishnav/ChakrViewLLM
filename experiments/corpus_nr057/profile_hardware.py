"""
Hardware Profiling Script for NR-057.
Performs 50-step disposable forward/backward passes on:
1. ChakrGenesis17M (d_model=384, layers=10, heads=12, kv=3, FFN=1024)
2. ChakrGenesis38M (d_model=512, layers=12, heads=8, kv=4, FFN=1408)

Measures:
- Peak VRAM allocation (GB)
- Step latency (ms)
- Throughput (tokens/second)
- Context length 512, batch size 8 and 16
- Autocast fp16 / bf16
- Checkpoint write latency
DOES NOT save trained weights or modify canonical models.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig

def profile_model(name: str, model_cls, config, batch_size: int, seq_len: int, steps: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Profiling {name} on {device} (Batch={batch_size}, SeqLen={seq_len}, Steps={steps}) ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Initialize fresh disposable model
    model = model_cls(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Generate dummy input tokens
    vocab_size = config.vocab_size
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(dummy_input)
            if isinstance(out, tuple):
                logits = out[0]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            loss = loss_fn(logits.view(-1, vocab_size), dummy_target.view(-1))
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    
    for step in range(steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(dummy_input)
            if isinstance(out, tuple):
                logits = out[0]
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            loss = loss_fn(logits.view(-1, vocab_size), dummy_target.view(-1))
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    step_latency_ms = (elapsed / steps) * 1000
    tokens_processed = batch_size * seq_len * steps
    tokens_per_sec = tokens_processed / elapsed
    
    # Checkpoint save time test (disposable)
    chkpt_path = f"experiments/corpus_nr057/temp_profile_{name}.pt"
    t_chkpt_start = time.perf_counter()
    torch.save(model.state_dict(), chkpt_path)
    chkpt_elapsed = (time.perf_counter() - t_chkpt_start) * 1000
    chkpt_size_mb = os.path.getsize(chkpt_path) / (1024 * 1024)
    if os.path.exists(chkpt_path):
        os.remove(chkpt_path)
        
    del model, optimizer
    torch.cuda.empty_cache()
    
    res = {
        "model_name": name,
        "parameters": param_count,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "total_time_seconds": round(elapsed, 3),
        "step_latency_ms": round(step_latency_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "checkpoint_save_ms": round(chkpt_elapsed, 2),
        "checkpoint_size_mb": round(chkpt_size_mb, 2)
    }
    print(f"Result for {name}: {res['tokens_per_sec']} tokens/s, {res['step_latency_ms']} ms/step, Peak VRAM: {res['peak_vram_gb']} GB")
    return res

def run_hardware_profile():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available for profiling")
        
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU: {gpu_name} ({gpu_total_vram_gb:.2f} GB total VRAM)")
    
    cfg17 = Genesis17MConfig()
    cfg38 = Genesis38MConfig()
    
    results = {
        "mission": "NR-057",
        "gpu_name": gpu_name,
        "total_vram_gb": round(gpu_total_vram_gb, 2),
        "profiles": []
    }
    
    # Profile 17M: batch 16, seq 512
    p17 = profile_model("ChakrGenesis17M", ChakrGenesis17M, cfg17, batch_size=16, seq_len=512, steps=50)
    results["profiles"].append(p17)
    
    # Profile 38M: batch 8 and batch 16, seq 512
    p38_b8 = profile_model("ChakrGenesis38M", ChakrGenesis38M, cfg38, batch_size=8, seq_len=512, steps=50)
    results["profiles"].append(p38_b8)
    
    p38_b16 = profile_model("ChakrGenesis38M", ChakrGenesis38M, cfg38, batch_size=16, seq_len=512, steps=50)
    results["profiles"].append(p38_b16)
    
    # Calculate estimated training times for 1 full epoch over Corpus V5 Train (3,187,865 tokens)
    corpus_v5_tokens = 3187865
    for p in results["profiles"]:
        tok_s = p["tokens_per_sec"]
        epoch_seconds = corpus_v5_tokens / tok_s
        p["estimated_epoch_minutes_corpus_v5"] = round(epoch_seconds / 60, 2)
        p["estimated_50m_tokens_hours"] = round((50000000 / tok_s) / 3600, 2)
        
    out_path = "experiments/corpus_nr057/hardware_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nHardware profiling complete and saved to {out_path}")

if __name__ == "__main__":
    run_hardware_profile()
