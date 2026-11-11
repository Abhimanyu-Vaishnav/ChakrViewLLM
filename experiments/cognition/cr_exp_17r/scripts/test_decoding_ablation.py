import json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

RELEASE_CKPT = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"

# Sample 10 prompts across diverse categories to evaluate decoding configs
PROMPTS = [
    "User: Hello! How can you help me today?\nAssistant:",
    "User: What is your official name?\nAssistant:",
    "User: Namaste! Aap kaise hain?\nAssistant:",
    "User: Can you assist me with Python code?\nAssistant:",
    "User: Tell me about your architecture.\nAssistant:",
    "User: What is the capital of Japan?\nAssistant:",
    "User: Thank you very much for your advice.\nAssistant:",
    "User: I live in Mumbai and work in tech.\nAssistant:",
    "User: Are you running on a cloud GPU?\nAssistant:",
    "User: Goodbye, see you tomorrow!\nAssistant:"
]

def test_decoding():
    runner = DirectNeuralRunner(RELEASE_CKPT)
    tok = runner.tokenizer
    
    configs = {
        "Greedy (rep_pen=1.0, no_repeat=0)": SamplingConfig(temperature=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0),
        "Excessive Penalty (rep_pen=1.25, no_repeat=3)": SamplingConfig(temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=3),
        "Calibrated Release (rep_pen=1.15, no_repeat=3)": SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3),
    }
    
    results = {}
    for cfg_name, cfg in configs.items():
        results[cfg_name] = []
        for p in PROMPTS:
            toks = tok.encode(p, add_bos=True, add_eos=False)
            curr = list(toks)
            gen = []
            with torch.no_grad():
                for step in range(40):
                    t_in = torch.tensor([curr], dtype=torch.long)
                    logits = runner.model(t_in)[0, -1, :]
                    next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                    gen.append(next_tok)
                    curr.append(next_tok)
                    if next_tok == 1:
                        break
            resp = tok.decode(gen, errors="replace").strip()
            results[cfg_name].append({"prompt": p.replace('\n', ' '), "response": resp})
            
    with open(REPO_ROOT / "experiments/cognition/cr_exp_17r/results/decoding_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("Decoding ablation evaluation complete.")
    for cfg_name, res in results.items():
        print(f"\n--- {cfg_name} ---")
        for r in res[:3]:
            print("P:", r["prompt"])
            print("R:", repr(r["response"]))

if __name__ == "__main__":
    test_decoding()
