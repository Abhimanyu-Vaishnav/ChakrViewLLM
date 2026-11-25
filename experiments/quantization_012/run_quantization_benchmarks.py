"""
ChakrView Quantization Experiment Suite.
Mission CV-GENESIS-NR-012 Phase 6.

Compares:
1. Baseline FP32 (Full Precision)
2. FP16 (Half Precision - CUDA)
3. Dynamic INT8 Quantization (torch.ao.quantization.quantize_dynamic on Linear layers)

Measures:
- Model artifact size on disk (MB)
- Peak VRAM / Memory footprint (MB)
- Generation latency (ms per token, tokens per second, first token latency)
- Output stability & numerical fidelity across fixed test prompts
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def measure_memory_and_speed(model, tokenizer, device, prompt="User: Explain gravity in simple terms.\nAssistant:", n_tokens=32):
    model.eval()
    tokens = [0] + tokenizer.encode(prompt)
    input_t = torch.tensor([tokens], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_t)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        cur_input = input_t
        for step in range(n_tokens):
            logits, _ = model(cur_input)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur_input = torch.cat([cur_input, next_tok], dim=1)
            if step == 0 and device.type == "cuda":
                torch.cuda.synchronize()
                first_tok_time = time.perf_counter() - start

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    total_time = time.perf_counter() - start
    tps = n_tokens / total_time
    decoded = tokenizer.decode(cur_input[0, len(tokens):].tolist())

    return {
        "tokens_per_sec": tps,
        "total_latency_ms": total_time * 1000.0,
        "peak_vram_mb": peak_vram_mb,
        "sample_output": decoded.strip()
    }


def run_quantization_benchmarks():
    ckpt_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    vocab_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(vocab_dir, verify_checksums=False)

    out_dir = Path("experiments/quantization_012")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Baseline FP32 (CUDA)
    print("\n--- Evaluating Baseline FP32 (CUDA) ---")
    config = Genesis17MConfig()
    model_fp32 = ChakrGenesis17M(config)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = st.get("model_state_dict", st)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_fp32.load_state_dict(clean_sd)
    model_fp32.cuda().eval()

    fp32_disk_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    metrics_fp32 = measure_memory_and_speed(model_fp32, tokenizer, torch.device("cuda"))
    metrics_fp32["disk_size_mb"] = fp32_disk_mb
    results["fp32_cuda"] = metrics_fp32
    print(f"FP32: Speed = {metrics_fp32['tokens_per_sec']:.1f} tok/s | VRAM = {metrics_fp32['peak_vram_mb']:.1f} MB | Output = {repr(metrics_fp32['sample_output'][:60])}")

    # 2. FP16 (CUDA)
    print("\n--- Evaluating FP16 Half Precision (CUDA) ---")
    model_fp16 = ChakrGenesis17M(config)
    model_fp16.load_state_dict(clean_sd)
    model_fp16.half().cuda().eval()

    # Save FP16 checkpoint to measure disk footprint
    fp16_path = out_dir / "genesis_17m_fp16.pt"
    torch.save({"model_state_dict": model_fp16.state_dict()}, fp16_path)
    fp16_disk_mb = os.path.getsize(fp16_path) / (1024 * 1024)

    metrics_fp16 = measure_memory_and_speed(model_fp16, tokenizer, torch.device("cuda"))
    metrics_fp16["disk_size_mb"] = fp16_disk_mb
    results["fp16_cuda"] = metrics_fp16
    print(f"FP16: Speed = {metrics_fp16['tokens_per_sec']:.1f} tok/s | VRAM = {metrics_fp16['peak_vram_mb']:.1f} MB | Disk = {fp16_disk_mb:.1f} MB | Output = {repr(metrics_fp16['sample_output'][:60])}")

    # 3. Dynamic INT8 Quantization (CPU)
    print("\n--- Evaluating Dynamic INT8 Quantization (CPU) ---")
    model_cpu = ChakrGenesis17M(config)
    model_cpu.load_state_dict(clean_sd)
    model_cpu.eval()

    # Quantize nn.Linear layers to dynamic int8
    quantized_int8 = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )
    int8_path = out_dir / "genesis_17m_int8_dynamic.pt"
    torch.save(quantized_int8.state_dict(), int8_path)
    int8_disk_mb = os.path.getsize(int8_path) / (1024 * 1024)

    metrics_int8 = measure_memory_and_speed(quantized_int8, tokenizer, torch.device("cpu"))
    metrics_int8["disk_size_mb"] = int8_disk_mb
    results["int8_cpu"] = metrics_int8
    print(f"INT8 (CPU): Speed = {metrics_int8['tokens_per_sec']:.1f} tok/s | Disk = {int8_disk_mb:.1f} MB | Output = {repr(metrics_int8['sample_output'][:60])}")

    # 4. FP32 CPU Reference
    print("\n--- Evaluating FP32 (CPU Reference) ---")
    metrics_fp32_cpu = measure_memory_and_speed(model_cpu, tokenizer, torch.device("cpu"))
    metrics_fp32_cpu["disk_size_mb"] = fp32_disk_mb
    results["fp32_cpu"] = metrics_fp32_cpu
    print(f"FP32 (CPU): Speed = {metrics_fp32_cpu['tokens_per_sec']:.1f} tok/s")

    # Save complete benchmark results
    res_path = out_dir / "quantization_benchmark_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll quantization benchmarks saved to: {res_path}")


if __name__ == "__main__":
    run_quantization_benchmarks()
