"""
Controlled Inference Diagnostics Script.
Tests CBR-SFT-07 Step 50 under 5 decoding configurations across key prompts.
Also performs logit diagnostic at token step 1 and step 2.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from chakrview.runtime.sampling import SamplingConfig
from scripts.run_m3_evaluations import DirectNeuralRunner

FORENSICS_DIR = REPO_ROOT / "experiments" / "cbr01_sft" / "conversational_forensics_01"
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_07" / "checkpoints" / "sft_checkpoint_step_50.pt"

PROMPTS = [
    ("What is your name?", "User: What is your name?\nAssistant:"),
    ("What can you do?", "User: What can you do?\nAssistant:"),
    ("My name is Abhimanyu.", "User: My name is Abhimanyu.\nAssistant:"),
    ("What is my name? (no prior turn)", "User: What is my name?\nAssistant:"),
    ("Explain neural networks.", "User: Explain neural networks.\nAssistant:"),
    ("Multi-turn Name Recall", "User: My name is Abhimanyu.\nAssistant: Hello Abhimanyu! Nice to meet you.\nUser: What is my name?\nAssistant:"),
]

DECODING_CONFIGS = {
    "A_Current_Chat": SamplingConfig(temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=3),
    "B_Greedy_Pure": SamplingConfig(temperature=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0),
    "C_Rep_Penalty_Only": SamplingConfig(temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=0),
    "D_Ngram_Only": SamplingConfig(temperature=0.0, repetition_penalty=1.0, no_repeat_ngram_size=3),
    "E_Sampling": SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1, no_repeat_ngram_size=0),
}

def generate_tokens(runner, prompt_str: str, config: SamplingConfig, max_new_tokens: int = 40):
    p_toks = runner.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    curr = list(p_toks)
    gen = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            t_in = torch.tensor([curr], dtype=torch.long)
            logits = runner.model(t_in)[0, -1, :]
            next_tok = runner.sampler.sample(
                logits,
                generated_tokens=gen,
                config=config,
                step=step
            )
            gen.append(next_tok)
            curr.append(next_tok)
            if next_tok == 1:
                break
                
    text = runner.tokenizer.decode(gen, errors="replace").strip()
    return text, gen

def get_top_logits(runner, prompt_str: str, top_k: int = 10):
    p_toks = runner.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    curr = list(p_toks)
    
    step_data = []
    with torch.no_grad():
        for step in range(2):
            t_in = torch.tensor([curr], dtype=torch.long)
            logits = runner.model(t_in)[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            top_vals, top_inds = torch.topk(probs, top_k)
            step_entries = []
            for val, ind in zip(top_vals.tolist(), top_inds.tolist()):
                tok_str = runner.tokenizer.decode([ind])
                step_entries.append({
                    "id": ind,
                    "token": tok_str,
                    "prob": round(val, 4),
                    "logit": round(logits[ind].item(), 3)
                })
            step_data.append(step_entries)
            
            # Take argmax to advance to step 2
            best_tok = top_inds[0].item()
            curr.append(best_tok)
            if best_tok == 1:
                break
                
    return step_data

def main():
    print(f"Loading model from {CHECKPOINT_PATH.name}...")
    runner = DirectNeuralRunner(CHECKPOINT_PATH)
    
    # 1. Decoding Comparison
    decoding_results = []
    for label, prompt in PROMPTS:
        row = {"label": label, "prompt": prompt, "generations": {}}
        for cfg_name, cfg in DECODING_CONFIGS.items():
            text, toks = generate_tokens(runner, prompt, cfg)
            row["generations"][cfg_name] = text
        decoding_results.append(row)
        
    md_decoding = FORENSICS_DIR / "DECODING_COMPARISON.md"
    with open(md_decoding, "w", encoding="utf-8") as f:
        f.write("# Decoding Configuration Diagnostic Matrix\n\n")
        f.write("Model: `CBR-SFT-07 Step 50`\n\n")
        for row in decoding_results:
            f.write(f"### Prompt: `{row['label']}`\n\n")
            f.write(f"**Raw Input**: ```\n{row['prompt']}\n```\n\n")
            f.write("| Configuration | Generated Response |\n|---|---|\n")
            for cfg_name, text in row["generations"].items():
                disp = text.replace("\n", " ")
                f.write(f"| `{cfg_name}` | {disp} |\n")
            f.write("\n")
    print(f"DECODING_COMPARISON.md written to {md_decoding}")
    
    # 2. Logit Analysis at Step 1 and Step 2
    logit_results = []
    for label, prompt in PROMPTS:
        top_data = get_top_logits(runner, prompt)
        logit_results.append({"label": label, "prompt": prompt, "steps": top_data})
        
    md_logits = FORENSICS_DIR / "LOGIT_ANALYSIS.md"
    with open(md_logits, "w", encoding="utf-8") as f:
        f.write("# Token-Level Logit and Probability Forensics (Step 1 & 2)\n\n")
        f.write("Model: `CBR-SFT-07 Step 50`\n\n")
        for entry in logit_results:
            f.write(f"## Prompt: `{entry['label']}`\n\n")
            for step_idx, step_entries in enumerate(entry["steps"], 1):
                f.write(f"### Next-Token Candidates (Step {step_idx})\n\n")
                f.write("| Rank | Token ID | Token String | Probability | Logit |\n|---|---|---|---|---|\n")
                for rank, c in enumerate(step_entries, 1):
                    f.write(f"| {rank} | {c['id']} | `{repr(c['token'])}` | {c['prob']:.4f} | {c['logit']:.3f} |\n")
                f.write("\n")
                
        f.write("## 3. Logit Forensic Synthesis\n\n")
        f.write("1. **Multi-turn Context Dissolution**: Under multi-turn prompt (`'User: My name is Abhimanyu... User: What is my name?'`), does the model attend to `'Abhimanyu'`? The logits reveal whether the top probability mass is concentrated on `'Abhimanyu'`, `'I'`, `'Your'`, or collapses to the refusal sink `'The'`/`'I do not'`.\n")
        f.write("2. **Entropy and Uncertainty**: Compare entropy in single-turn identity queries vs multi-turn context queries.\n")

    print(f"LOGIT_ANALYSIS.md written to {md_logits}")

if __name__ == "__main__":
    main()

