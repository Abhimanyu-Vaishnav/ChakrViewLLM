import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig
from chakrview.chat import ConversationalChatCLI

RELEASE_CKPT = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"
CONTEXT_BENCH = REPO_ROOT / "experiments/cognition/cr_exp_19/benchmark/CR19_CONTEXT_200.json"
CONV_BENCH = REPO_ROOT / "experiments/cognition/cr_exp_19/benchmark/CR19_CONVERSATION_100.json"
RESULTS_DIR = REPO_ROOT / "experiments/cognition/cr_exp_19/results"

def evaluate_baseline():
    runner = DirectNeuralRunner(RELEASE_CKPT)
    tok = runner.tokenizer
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    
    # Initialize CLI for path B
    cli = ConversationalChatCLI(RELEASE_CKPT)
    
    with open(CONTEXT_BENCH, encoding="utf-8") as f:
        ctx_data = json.load(f)
    with open(CONV_BENCH, encoding="utf-8") as f:
        conv_data = json.load(f)
        
    print("Evaluating Path A (Direct Full History) on CR19_CONTEXT_200 (Sample 40 probes, 8 per group)...")
    sample_ctx = []
    for g in ["1-turn", "2-turn", "3-turn", "5-turn", "8-turn"]:
        items = [x for x in ctx_data if x["group"] == g]
        sample_ctx.extend(items[:8])
        
    path_a_results = []
    path_b_results = []
    
    for p in sample_ctx:
        prompt_full = p["formatted_prompt"]
        exp = p["expected"]
        
        # Path A: Direct model inference with full prompt
        toks = tok.encode(prompt_full, add_bos=True, add_eos=False)
        curr = list(toks)
        gen_a = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen_a, config=cfg, step=step)
                gen_a.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    break
        resp_a = tok.decode(gen_a, errors="replace").strip()
        hit_a = any(e.lower() in resp_a.lower() for e in exp)
        path_a_results.append({
            "id": p["id"],
            "group": p["group"],
            "response": resp_a,
            "hit": hit_a
        })
        
        # Path B: Corrected CLI runtime prompt packing
        cli.reset()
        for t in p["turns"][:-1]:
            cli.history.append({"user": t["user"], "assistant": t["assistant"]})
        last_user = p["turns"][-1]["user"]
        p_b, _ = cli.build_prompt(last_user)
        
        toks_b = tok.encode(p_b, add_bos=True, add_eos=False)
        curr_b = list(toks_b)
        gen_b = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr_b], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen_b, config=cfg, step=step)
                gen_b.append(tok_id)
                curr_b.append(tok_id)
                if tok_id == 1:
                    break
        resp_b = tok.decode(gen_b, errors="replace").strip()
        hit_b = any(e.lower() in resp_b.lower() for e in exp)
        path_b_results.append({
            "id": p["id"],
            "group": p["group"],
            "response": resp_b,
            "hit": hit_b
        })
        
    # Evaluate CR19_CONVERSATION_100 (Sample 30 probes)
    print("Evaluating baseline on CR19_CONVERSATION_100...")
    conv_results = []
    for p in conv_data[:30]:
        prompt = p["prompt"]
        exp = p["expected"]
        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    break
        resp = tok.decode(gen, errors="replace").strip()
        hit = any(e.lower() in resp.lower() for e in exp)
        conv_results.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": prompt,
            "response": resp,
            "hit": hit
        })
        
    summary = {
        "checkpoint": "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt",
        "checkpoint_sha256": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10",
        "path_a_full_history_hits": sum(x["hit"] for x in path_a_results),
        "path_a_total": len(path_a_results),
        "path_b_cli_runtime_hits": sum(x["hit"] for x in path_b_results),
        "path_b_total": len(path_b_results),
        "conversation_preservation_hits": sum(x["hit"] for x in conv_results),
        "conversation_preservation_total": len(conv_results),
        "path_a_details": path_a_results,
        "path_b_details": path_b_results,
        "conv_details": conv_results
    }
    with open(RESULTS_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Baseline Eval Summary: Path A Hits: {summary['path_a_full_history_hits']}/{summary['path_a_total']}")
    print(f"Path B Hits: {summary['path_b_cli_runtime_hits']}/{summary['path_b_total']}")
    print(f"Conversation Hits: {summary['conversation_preservation_hits']}/{summary['conversation_preservation_total']}")

if __name__ == "__main__":
    evaluate_baseline()

