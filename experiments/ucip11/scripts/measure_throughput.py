"""
Throughput Benchmark and Training Capacity Measurement for Phase F.
Benchmarks:
1. Autoregressive training step throughput on CPU (samples/sec, tokens/sec)
2. Evaluation throughput (generation tokens/sec)
3. Memory and CPU utilization
"""

import sys
import time
import psutil
import torch
from torch.optim import AdamW
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from scripts.run_m3_evaluations import DirectNeuralRunner

print("Measuring training and evaluation throughput on CPU...")

# 1. 12.9M Medium model benchmark
medium_ckpt = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
runner = DirectNeuralRunner(medium_ckpt)
model = runner.model
model.train()

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

batch_size = 4
seq_len = 128
dummy_inp = torch.randint(0, 4096, (batch_size, seq_len), dtype=torch.long)
dummy_tgt = torch.randint(0, 4096, (batch_size, seq_len), dtype=torch.long)

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    logits = model(dummy_inp)
    loss = criterion(logits[:, :-1, :].reshape(-1, 4096), dummy_tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

# Timed run: 10 steps
t0 = time.time()
steps = 10
for _ in range(steps):
    optimizer.zero_grad()
    logits = model(dummy_inp)
    loss = criterion(logits[:, :-1, :].reshape(-1, 4096), dummy_tgt[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
t1 = time.time()

elapsed = t1 - t0
tokens_processed = steps * batch_size * seq_len
tok_per_sec = tokens_processed / elapsed
sec_per_step = elapsed / steps

mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
cpu_pct = psutil.cpu_percent(interval=0.5)

print(f"\n--- Medium Model (12.9M Parameters) ---")
print(f"Elapsed for {steps} steps: {elapsed:.2f}s ({sec_per_step:.2f} s/step)")
print(f"Throughput: {tok_per_sec:.1f} tokens/second")
print(f"RAM Usage: {mem_mb:.1f} MB | Host CPU Utilization: {cpu_pct:.1f}%")

# 2. Generation Throughput
model.eval()
gen_t0 = time.time()
with torch.no_grad():
    curr = [0, 45, 12, 89]
    for _ in range(32):
        t_in = torch.tensor([curr], dtype=torch.long)
        logits = model(t_in)[0, -1, :]
        nxt = int(torch.argmax(logits).item())
        curr.append(nxt)
gen_t1 = time.time()
gen_tok_per_sec = 32 / (gen_t1 - gen_t0)
print(f"Generation Throughput: {gen_tok_per_sec:.1f} tokens/second")
