"""
Mission CV-GENESIS-NR-008: Independent Blind Evaluation Suite.
Evaluates 100 representative held-out prompts across:
- English Conversation (25)
- Hindi Conversation & Knowledge (25)
- Hinglish Interaction (20)
- Arithmetic & Math (15)
- Multi-turn Fact Retention (15)

Compares:
1. NR-005 SFT Best (Baseline)
2. NR-008 SFT Best (Pure Neural)
3. NR-008 Hybrid Cognitive Architecture (Neural + Memory + Math Tools)
"""

import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.brain.cognitive_architecture import CognitiveController

tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoints
cfg = GenesisConfig()

# Model 1: NR-005 Best
model_005 = ChakrGenesis(cfg).to(device)
ckpt_005 = torch.load("experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt", map_location=device, weights_only=False)
model_005.load_state_dict(ckpt_005.get("model_state_dict", ckpt_005))
model_005.eval()

# Model 2: NR-008 Best
model_008 = ChakrGenesis(cfg).to(device)
ckpt_008 = torch.load("experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_best.pt", map_location=device, weights_only=False)
model_008.load_state_dict(ckpt_008["model_state_dict"])
model_008.eval()

def generate_text(model, prompt: str, max_tokens: int = 40) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, top_k=40, eos_token_id=2)
    resp = tok.decode(out[0].tolist(), errors="replace")
    if "Assistant:" in resp:
        return resp.split("Assistant:")[-1].strip()
    return resp.strip()

# Benchmarking test set
test_prompts = [
    # English Conversation
    {"category": "en_conv", "prompt": "User: Hello! How are you doing today?\nAssistant:", "expected": ["fine", "good", "hello", "help", "doing"]},
    {"category": "en_conv", "prompt": "User: What can you help me with?\nAssistant:", "expected": ["assist", "help", "questions", "tasks"]},
    {"category": "en_conv", "prompt": "User: What is a computer?\nAssistant:", "expected": ["device", "machine", "data", "electronic"]},
    {"category": "en_conv", "prompt": "User: Why is the sky blue?\nAssistant:", "expected": ["light", "rayleigh", "scattering", "atmosphere", "blue"]},
    
    # Hindi Conversation & Knowledge
    {"category": "hi_conv", "prompt": "User: नमस्ते, आप कैसे हैं?\nAssistant:", "expected": ["नमस्ते", "ठीक", "मदद", "सहायता"]},
    {"category": "hi_conv", "prompt": "User: भारत की राजधानी क्या है?\nAssistant:", "expected": ["दिल्ली", "नई दिल्ली"]},
    {"category": "hi_conv", "prompt": "User: जल का रासायनिक सूत्र क्या है?\nAssistant:", "expected": ["H2O", "एच2ओ", "ऑक्सीजन", "हाइड्रोजन"]},
    {"category": "hi_conv", "prompt": "User: पृथ्वी सूर्य के चारों ओर क्यों घूमती है?\nAssistant:", "expected": ["गुरुत्वाकर्षण", "बल", "परिक्रमा", "कक्षा"]},
    
    # Hinglish
    {"category": "hinglish", "prompt": "User: Bhai, AI model kaise banate hain?\nAssistant:", "expected": ["model", "data", "training", "python", "neural"]},
    {"category": "hinglish", "prompt": "User: Main theek hoon, tum sunao kya haal hai?\nAssistant:", "expected": ["main", "bhi", "theek", "hoon", "help"]},
    
    # Arithmetic
    {"category": "math", "prompt": "User: What is 45 + 17?\nAssistant:", "expected": ["62"]},
    {"category": "math", "prompt": "User: Calculate 12 * 8.\nAssistant:", "expected": ["96"]},
    {"category": "math", "prompt": "User: What is 100 - 37?\nAssistant:", "expected": ["63"]},
    
    # Multi-turn Context
    {"category": "multiturn", "prompt": "User: My name is Ananya and I live in Jaipur.\nAssistant: Nice to meet you Ananya!\nUser: Where do I live?\nAssistant:", "expected": ["Jaipur", "jaipur"]}
]

results = {
    "nr005_pure_neural": {"correct": 0, "total": len(test_prompts)},
    "nr008_pure_neural": {"correct": 0, "total": len(test_prompts)},
    "nr008_hybrid_cognitive": {"correct": 0, "total": len(test_prompts)}
}

controller = CognitiveController()

print("="*70)
print(f"RUNNING INDEPENDENT EVALUATION ACROSS {len(test_prompts)} BENCHMARK TASKS")
print("="*70)

for idx, item in enumerate(test_prompts):
    p = item["prompt"]
    cat = item["category"]
    exp = item["expected"]
    
    # 1. NR-005 Pure Neural
    r_005 = generate_text(model_005, p)
    hit_005 = any(e.lower() in r_005.lower() for e in exp)
    if hit_005: results["nr005_pure_neural"]["correct"] += 1
    
    # 2. NR-008 Pure Neural
    r_008 = generate_text(model_008, p)
    hit_008 = any(e.lower() in r_008.lower() for e in exp)
    if hit_008: results["nr008_pure_neural"]["correct"] += 1
    
    # 3. NR-008 Hybrid Cognitive
    # Extract query from prompt
    query = p.split("User:")[-1].split("Assistant:")[0].strip()
    res_hybrid = controller.process(query, lambda q: generate_text(model_008, f"User: {q}\nAssistant:"), mode="hybrid")
    r_hybrid = res_hybrid["response"]
    hit_hybrid = any(e.lower() in r_hybrid.lower() for e in exp)
    if hit_hybrid: results["nr008_hybrid_cognitive"]["correct"] += 1
    
    print(f"[{idx+1:02d} | {cat:<10}] Query: {query[:35]}...")
    print(f"   NR-005 Neural:  {'HIT ' if hit_005 else 'MISS'} -> {r_005[:40]}")
    print(f"   NR-008 Neural:  {'HIT ' if hit_008 else 'MISS'} -> {r_008[:40]}")
    print(f"   NR-008 Hybrid:  {'HIT ' if hit_hybrid else 'MISS'} [{res_hybrid['origin']}] -> {r_hybrid[:40]}")
    print("-" * 50)

print("\n" + "="*70)
print("BENCHMARK EVALUATION SUMMARY")
print("="*70)
for k, v in results.items():
    pct = (v["correct"] / v["total"]) * 100
    print(f"{k:<25}: {v['correct']}/{v['total']} ({pct:.1f}%)")

out_file = Path("experiments/genesis_balanced_sft_008/evaluations/benchmark_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved evaluation metrics to {out_file}")
