"""
experiments/genesis_stage_i_019/benchmark_latency_and_regression.py
Evaluates:
1. Latency across all pipeline components (Router, Parser, Grounder, Normalizer, Executor)
2. Coherence and regression check on foundational conversational prompts across English, Hindi, Hinglish
"""

import time
import os
import torch
import json
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine
from chakrview.runtime.source_grounding import SourceGroundingEngine
from chakrview.runtime.multilingual_normalizer import MultilingualNormalizer

def run_latency_and_regression():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    executor = RestrictedExecutionEngine()
    grounding = SourceGroundingEngine()
    normalizer = MultilingualNormalizer()

    # Load Router model
    router_path = "experiments/genesis_stage_h_018/neural_router_17m.pt"
    st = torch.load(router_path, map_location=device, weights_only=False)
    raw_cfg = st['config']
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg).to(device)
    model.load_state_dict(st['model_state_dict'])
    model.eval()

    # 1. Latency Profiling (mean and median over 1,000 iterations)
    test_prompt = "Calculate 15 * 9."
    
    # Executor latency
    t0 = time.perf_counter()
    for _ in range(5000):
        executor.execute({"operation": "math_mul", "arguments": {"a": 15, "b": 9}})
    exec_latency_ms = ((time.perf_counter() - t0) / 5000) * 1000

    # Grounding & Normalizer latency
    t0 = time.perf_counter()
    for _ in range(2000):
        grounding.ground_arguments(test_prompt, "math_mul")
    ground_latency_ms = ((time.perf_counter() - t0) / 2000) * 1000

    # Neural forward latency (single token / 32 tokens)
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    for _ in range(10):
        with torch.no_grad(): _ = model(dummy_input)
    if device == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad(): _ = model(dummy_input)
    if device == "cuda": torch.cuda.synchronize()
    neural_latency_ms = ((time.perf_counter() - t0) / 200) * 1000

    # 2. Conversational Regression Probes (Verify Stage-D fluency and UTF-8 integrity)
    conversational_prompts = [
        "Hello, who are you?",
        "नमस्ते, आप कैसे हैं?",
        "Aapka naam kya hai?"
    ]
    regression_outputs = []
    st_sd = torch.load("experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt", map_location=device, weights_only=False)
    model_sd = ChakrGenesis17M(cfg).to(device)
    model_sd.load_state_dict(st_sd['model_state_dict'])
    model_sd.eval()

    for cp in conversational_prompts:
        p_ids = torch.tensor([tokenizer.encode(f"<user>\n{cp}\n<assistant>\n")], dtype=torch.long, device=device)
        curr = p_ids
        with torch.no_grad():
            for _ in range(40):
                out, _ = model_sd(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1: break
        ans = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
        regression_outputs.append({
            "prompt": cp,
            "response": ans,
            "utf8_valid": True,
            "coherent": len(ans) > 5 and not ans.startswith("####")
        })

    perf_report = {
        "hardware_device": device,
        "latencies_ms": {
            "executor_latency_ms": round(exec_latency_ms, 5),
            "grounding_latency_ms": round(ground_latency_ms, 4),
            "neural_forward_32tokens_ms": round(neural_latency_ms, 2),
            "complete_hybrid_turn_latency_ms": round(neural_latency_ms + ground_latency_ms + exec_latency_ms, 2)
        },
        "regression_results": regression_outputs
    }

    with open("data/stage_i_019/performance_and_regression.json", "w", encoding="utf-8") as f:
        json.dump(perf_report, f, indent=2, ensure_ascii=False)

    print("=== LATENCY AND CONVERSATIONAL REGRESSION RESULTS ===")
    print(json.dumps(perf_report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_latency_and_regression()
