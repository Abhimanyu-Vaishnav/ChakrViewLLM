import time
import psutil
import os
import sys
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.chat import ConversationalChatCLI

def profile_performance():
    ckpt_path = REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt"
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    t0 = time.perf_counter()
    cli = ConversationalChatCLI(ckpt_path)
    load_time = time.perf_counter() - t0
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    # Measure generation tokens and speed
    prompts = [
        "Hello assistant, tell me about your project.",
        "Mera naam Priya hai aur main Pune me rehti hoon. Aapka naam kya hai?",
        "Can you explain what a transformer neural network does?"
    ]
    
    total_tokens = 0
    total_time = 0.0
    
    for p in prompts:
        t_start = time.perf_counter()
        resp = cli.interact_step(p)
        dt = time.perf_counter() - t_start
        toks = len(cli.tok.encode(resp))
        total_tokens += toks
        total_time += dt
        print(f"Prompt: '{p[:30]}...' -> Tokens: {toks}, Latency: {dt*1000:.1f}ms")
        
    avg_latency = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
    tps = total_tokens / total_time if total_time > 0 else 0
    peak_mem = process.memory_info().rss / (1024 * 1024)
    
    print("\n--- Hardware Resource Profile ---")
    print(f"Model Load Time:        {load_time:.2f}s")
    print(f"Memory (Pre-load):      {mem_before:.1f} MB")
    print(f"Memory (Post-load):     {mem_after:.1f} MB")
    print(f"Peak Memory:            {peak_mem:.1f} MB")
    print(f"Average Token Latency:  {avg_latency:.1f} ms/token")
    print(f"Generation Throughput:  {tps:.1f} tokens/sec")
    print(f"CPU Threads:            {torch.get_num_threads()}")

if __name__ == "__main__":
    profile_performance()
