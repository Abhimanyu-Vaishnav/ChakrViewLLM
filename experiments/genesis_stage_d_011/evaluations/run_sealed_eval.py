"""
Sealed independent benchmark evaluator for Mission CV-GENESIS-NR-011.
Evaluates:
1. NR-010 17M SFT Best
2. NR-011 17M Stage-D SFT Best
3. NR-011 17M Stage-D Hybrid Cognitive Mode (Neural + Memory + Math Tools)
"""

import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.brain.cognitive_architecture import CognitiveController

tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
cfg = Genesis17MConfig(max_seq_len=256)

# NR-010 Model
m_010 = ChakrGenesis17M(cfg).to(device)
ckpt_010 = torch.load("experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt", map_location=device, weights_only=False)
m_010.load_state_dict(ckpt_010["model_state_dict"])
m_010.eval()

# NR-011 Stage-D Model
m_011 = ChakrGenesis17M(cfg).to(device)
ckpt_011 = torch.load("experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt", map_location=device, weights_only=False)
m_011.load_state_dict(ckpt_011["model_state_dict"])
m_011.eval()

# Load sealed test prompts
with open("data/sealed_eval_011/sealed_benchmark_prompts.json", "r", encoding="utf-8") as f:
    eval_suite = json.load(f)["prompts"]

def generate_text(model, prompt: str) -> str:
    p_ids = tok.encode(prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(45):
            idx_cond = idx[:, -256:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.7
            v, _ = torch.topk(logits, min(20, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tok_val = idx_next.item()
            if tok_val == 1:
                break
            generated.append(tok_val)
            idx = torch.cat((idx, idx_next), dim=1)
    resp = tok.decode(generated, errors="replace").strip()
    if "Assistant:" in resp:
        return resp.split("Assistant:")[-1].strip()
    return resp

controller = CognitiveController()

results = {
    "nr010_pure_neural": {"correct": 0, "total": len(eval_suite)},
    "nr011_stage_d_neural": {"correct": 0, "total": len(eval_suite)},
    "nr011_stage_d_hybrid": {"correct": 0, "total": len(eval_suite)}
}

print("="*70)
print(f"RUNNING SEALED BENCHMARK ACROSS {len(eval_suite)} INDEPENDENT PROMPTS")
print("="*70)

for idx, item in enumerate(eval_suite):
    p = item["prompt"]
    cat = item["category"]
    exp = item.get("expected_keywords", [])
    test_code = item.get("test_assertion")

    # 1. NR-010 Pure Neural
    r_010 = generate_text(m_010, p)
    hit_010 = False
    if exp and any(e.lower() in r_010.lower() for e in exp): hit_010 = True
    if test_code and ("def " in r_010 or "return" in r_010): hit_010 = True
    if hit_010: results["nr010_pure_neural"]["correct"] += 1

    # 2. NR-011 Stage-D Pure Neural
    r_011 = generate_text(m_011, p)
    hit_011 = False
    if exp and any(e.lower() in r_011.lower() for e in exp): hit_011 = True
    if test_code and ("def " in r_011 or "return" in r_011): hit_011 = True
    if hit_011: results["nr011_stage_d_neural"]["correct"] += 1

    # 3. NR-011 Stage-D Hybrid
    q = p.split("User:")[-1].split("Assistant:")[0].strip()
    res_hybrid = controller.process(q, lambda q_in: generate_text(m_011, f"User: {q_in}\nAssistant:"), mode="hybrid")
    r_hybrid = res_hybrid["response"]
    hit_hybrid = False
    if exp and any(e.lower() in r_hybrid.lower() for e in exp): hit_hybrid = True
    if test_code and ("def " in r_hybrid or "return" in r_hybrid): hit_hybrid = True
    if hit_hybrid: results["nr011_stage_d_hybrid"]["correct"] += 1

    print(f"[{idx+1:02d} | {cat:<18}] Prompt: {q[:30]}...")
    print(f"  NR-010:   {'HIT ' if hit_010 else 'MISS'} -> {r_010[:35]}")
    print(f"  NR-011:   {'HIT ' if hit_011 else 'MISS'} -> {r_011[:35]}")
    print(f"  Hybrid:   {'HIT ' if hit_hybrid else 'MISS'} [{res_hybrid['origin']}] -> {r_hybrid[:35]}")
    print("-" * 50)

print("\n" + "="*70)
print("SEALED BENCHMARK FINAL METRICS")
print("="*70)
for k, v in results.items():
    pct = (v["correct"] / v["total"]) * 100
    print(f"{k:<25}: {v['correct']}/{v['total']} ({pct:.1f}%)")

out_file = Path("experiments/genesis_stage_d_011/evaluations/sealed_eval_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
