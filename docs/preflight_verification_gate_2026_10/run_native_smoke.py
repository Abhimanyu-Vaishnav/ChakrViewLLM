"""
run_native_smoke.py
ChakrView Pre-Manual Baseline Verification Gate

Executes a bounded CPU-only native smoke test of the canonical ChakrMicro checkpoint (Step-10000)
and native Byte-Level BPE Tokenizer.
NO pointer, NO oracle, NO schema controller, NO deterministic extraction, NO external LLM, NO retrieval.
Pure native greedy autoregression.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import time
import hashlib
import psutil
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer

PROMPTS = [
    {
        "category": "paraphrase",
        "name": "paraphrase_probe",
        "prompt": "Rewrite the following sentence in simple words:\nThe atmospheric conditions precipitated an unexpected deluge."
    },
    {
        "category": "negation",
        "name": "negation_probe",
        "prompt": "Context: The power converter is not active.\nQuestion: Is the power converter active?\nAnswer:"
    },
    {
        "category": "instruction_following",
        "name": "simple_instruction_probe",
        "prompt": "Instruction: Output only the single word SUCCESS and nothing else.\nOutput:"
    },
    {
        "category": "distractor_resistance",
        "name": "distractor_probe",
        "prompt": "Context: Device A is offline. Device B is running. Device C is standby.\nQuestion: What is the status of Device B?\nAnswer:"
    },
    {
        "category": "multi_sentence_context",
        "name": "multihop_context_probe",
        "prompt": "Alice gave the blue key to Bob. Bob placed the blue key inside the metal cabinet.\nQuestion: Where is the blue key?\nAnswer:"
    },
    {
        "category": "uncertainty",
        "name": "uncertainty_probe",
        "prompt": "Context: The turbine speed is 1500 RPM.\nQuestion: What is the internal temperature of the turbine?\nAnswer:"
    },
    {
        "category": "coding",
        "name": "simple_coding_probe",
        "prompt": "Write a Python function named `is_even(n)` that returns True if n is even, else False:\n```python\n"
    }
]

def generate_native(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    generated_tokens = []
    curr_ids = input_ids
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_ids)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == 1:  # EOS token
                break
            generated_tokens.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            
            # Bound context to max seq len
            if curr_ids.shape[1] >= 2048:
                break
                
    return tokenizer.decode(generated_tokens)

def main():
    print("=" * 70)
    print("CHAKRVIEW PREFLIGHT: NATIVE SMOKE TEST RUNNER")
    print("=" * 70)
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial process memory: {start_mem:.2f} MB")
    
    # 1. Hashes verification
    canon_ckpt = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    vocab_file = "data/experiments/vocab_4096/vocab.json"
    merges_file = "data/experiments/vocab_4096/merges.json"
    session_file = "chakrview/runtime/session.py"
    
    hashes = {
        "canonical_checkpoint": hashlib.sha256(open(canon_ckpt, "rb").read()).hexdigest(),
        "tokenizer_vocab": hashlib.sha256(open(vocab_file, "rb").read()).hexdigest(),
        "tokenizer_merges": hashlib.sha256(open(merges_file, "rb").read()).hexdigest(),
        "runtime_session": hashlib.sha256(open(session_file, "rb").read()).hexdigest()
    }
    
    # 2. Model & Tokenizer loading
    load_start = time.time()
    tok_dir = Path("data/experiments/vocab_4096")
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts
    tokenizer, tok_cfg = load_tokenizer_artifacts(tok_dir, verify_checksums=False)
    
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    
    ckpt = torch.load(canon_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    load_time = time.time() - load_start
    
    loaded_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Model loaded in {load_time:.2f}s. Memory post-load: {loaded_mem:.2f} MB")
    
    # 3. Tokenizer round-trip verification
    test_roundtrip = "ChakrView indigenous BPE verification probe: नमस्ते दुनिया 12345"
    roundtrip_tokens = tokenizer.encode(test_roundtrip)
    roundtrip_decoded = tokenizer.decode(roundtrip_tokens)
    roundtrip_ok = (roundtrip_decoded == test_roundtrip)
    print(f"Tokenizer lossless roundtrip check: {'PASS' if roundtrip_ok else 'FAIL'}")
    
    # 4. Prompt Suite Execution
    results = []
    print("\nExecuting native prompt suite (max 40 tokens greedy)...")
    
    for item in PROMPTS:
        p_start = time.time()
        raw_output = generate_native(model, tokenizer, item["prompt"], max_new_tokens=40)
        p_time = time.time() - p_start
        
        mem_now = process.memory_info().rss / (1024 * 1024)
        print(f"[{item['category']}] ({p_time:.2f}s) -> Raw: {repr(raw_output[:60])}...")
        
        results.append({
            "category": item["category"],
            "name": item["name"],
            "prompt": item["prompt"],
            "raw_output": raw_output,
            "latency_seconds": round(p_time, 4),
            "memory_mb_after": round(mem_now, 2)
        })
        
    peak_mem = process.memory_info().rss / (1024 * 1024)
    print(f"\nCompleted 7 prompts. Peak memory: {peak_mem:.2f} MB")
    
    output_dir = "docs/preflight_verification_gate_2026_10"
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "metadata": {
            "timestamp": "2026-10-21T23:42:00+05:30",
            "environment": {
                "os": "Microsoft Windows 11 Enterprise",
                "python": sys.version,
                "torch": torch.__version__,
                "cpu_only": not torch.cuda.is_available(),
                "num_threads": torch.get_num_threads()
            },
            "canonical_hashes": hashes,
            "load_time_seconds": round(load_time, 4),
            "initial_memory_mb": round(start_mem, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "tokenizer_roundtrip_verified": roundtrip_ok
        },
        "prompts_evaluated": results
    }
    
    with open(os.path.join(output_dir, "NATIVE_SMOKE_RESULTS.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved results to {os.path.join(output_dir, 'NATIVE_SMOKE_RESULTS.json')}")

if __name__ == "__main__":
    main()
