import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
import torch
from scripts.run_m3_evaluations import DirectNeuralRunner, file_sha256

checkpoints = [
    ("v0.11", REPO_ROOT / "releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt"),
    ("UCIP-03 Micro", REPO_ROOT / "releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt"),
    ("UCIP-04 Micro", REPO_ROOT / "releases/ucip_04_candidate/chakrview_ucip04_model.pt"),
    ("UCIP-04 Medium", REPO_ROOT / "releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt"),
    ("UCIP-01R 13M Parent", REPO_ROOT / "experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt")
]

test_prompts = [
    "Hello ChakrView. How are you?",
    "What does a CPU do?",
    "Who are you? Explain how you were created.",
    "नमस्ते! आप क्या कर सकते हैं?",
    "Bhai, machine learning ko simple language mein samjhao.",
    "Say exactly: The sky is blue.",
    "If I have 5 apples and give 2 to my friend, how many apples remain?"
]

results = {}

for label, ckpt_path in checkpoints:
    if not ckpt_path.exists():
        print(f"Skipping {label}: not found at {ckpt_path}")
        continue
    print(f"\nEvaluating Checkpoint: {label}")
    sha = file_sha256(ckpt_path)
    runner = DirectNeuralRunner(ckpt_path)
    arch = getattr(runner, "arch_name", "ChakrMicro")
    param_count = sum(p.numel() for p in runner.model.parameters())
    
    ckpt_res = {
        "sha256": sha,
        "architecture": arch,
        "parameters": param_count,
        "responses": []
    }
    
    for p in test_prompts:
        prompt_formatted = f"User: {p}\nAssistant:"
        out = runner.generate(prompt_formatted, max_new_tokens=48)
        resp_text = out.get("text", "").strip()
        tokens = out.get("tokens", [])
        ckpt_res["responses"].append({
            "prompt": p,
            "response": resp_text,
            "tokens": tokens,
            "stop_reason": out.get("stop_reason")
        })
        print(f"  [{label}] Q: {p[:30]}... -> {repr(resp_text[:60])}")
        
    results[label] = ckpt_res

with open(REPO_ROOT / "experiments/cognition/ucip_05/checkpoint_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nAudit comparison complete. Saved to checkpoint_comparison.json.")
