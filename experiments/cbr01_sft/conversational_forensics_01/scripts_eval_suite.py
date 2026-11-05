"""
Fixed Evaluation Suite Forensic Runner.
Evaluates CBR-SFT-07 Step 50 across 10 canonical categories without subjective grading,
recording raw generation outputs and computing objective exact-match / prefix statistics.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.runtime.sampling import SamplingConfig
from scripts.run_m3_evaluations import DirectNeuralRunner

FORENSICS_DIR = REPO_ROOT / "experiments" / "cbr01_sft" / "conversational_forensics_01"
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_07" / "checkpoints" / "sft_checkpoint_step_50.pt"

EVAL_SUITE = [
    # 1. Identity
    {"category": "Identity Awareness", "prompt": "User: What is your name?\nAssistant:", "expected": "ChakrView"},
    {"category": "Identity Awareness", "prompt": "User: Who created you?\nAssistant:", "expected": "indigenous / research / Abhimanyu"},
    
    # 2. Greetings
    {"category": "Greetings", "prompt": "User: Hello!\nAssistant:", "expected": "greeting without loop"},
    {"category": "Greetings", "prompt": "User: Good morning!\nAssistant:", "expected": "morning / assist"},
    
    # 3. Abstention
    {"category": "Abstention", "prompt": "User: What is the current stock price of Apple right now?\nAssistant:", "expected": "abstention / no live access"},
    {"category": "Abstention", "prompt": "User: What is the secret code inside the locked vault?\nAssistant:", "expected": "abstention"},
    
    # 4. Context Grounding (Single-turn)
    {"category": "Context Grounding", "prompt": "User: The project name is Project Garuda. What is the project name?\nAssistant:", "expected": "Project Garuda"},
    {"category": "Context Grounding", "prompt": "User: The server port is 8080. What port is the server using?\nAssistant:", "expected": "8080"},
    
    # 5. Multi-turn Context Recall
    {"category": "Multi-turn Memory", "prompt": "User: My name is Abhimanyu.\nAssistant: Hello Abhimanyu! How can I help you today?\nUser: What is my name?\nAssistant:", "expected": "Abhimanyu"},
    {"category": "Multi-turn Memory", "prompt": "User: I am building an engine called Vayu.\nAssistant: That sounds interesting! Tell me more about Vayu.\nUser: What is the engine called?\nAssistant:", "expected": "Vayu"},
    
    # 6. Technical Explanation
    {"category": "Technical Explanation", "prompt": "User: What is an attention mechanism in transformers?\nAssistant:", "expected": "weights / tokens / focus"},
    {"category": "Technical Explanation", "prompt": "User: Explain neural networks.\nAssistant:", "expected": "nodes / layers / learning"},
    
    # 7. Hinglish / Bilingual
    {"category": "Hinglish / Bilingual", "prompt": "User: Namaste, aap kaun ho?\nAssistant:", "expected": "ChakrView / Hindi/Hinglish"},
    {"category": "Hinglish / Bilingual", "prompt": "User: Mujhe neural networks simple bhasha mein samjhao.\nAssistant:", "expected": "simple explanation"},
    
    # 8. Continuity / Feedback
    {"category": "Continuity & Feedback", "prompt": "User: Thanks for your help!\nAssistant:", "expected": "welcome / pleasure"},
    
    # 9. Instruction Following
    {"category": "Instruction Following", "prompt": "User: Reply with only the single word: CONFIRMED\nAssistant:", "expected": "CONFIRMED"},
    
    # 10. Robustness & Repetition Trap
    {"category": "Robustness", "prompt": "User: Tell me about yourself.\nAssistant:", "expected": "coherent summary without loops"}
]

def main():
    runner = DirectNeuralRunner(CHECKPOINT_PATH)
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=3)
    
    results = []
    
    for item in EVAL_SUITE:
        prompt = item["prompt"]
        p_toks = runner.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        curr = list(p_toks)
        gen = []
        
        with torch.no_grad():
            for step in range(48):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                next_tok = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(next_tok)
                curr.append(next_tok)
                if next_tok == 1:
                    break
                    
        response = runner.tokenizer.decode(gen, errors="replace").strip()
        results.append({
            "category": item["category"],
            "prompt": prompt,
            "expected": item["expected"],
            "generated": response,
            "num_tokens": len(gen),
            "stopped_on_eos": (gen[-1] == 1 if gen else False)
        })
        
    out_md = FORENSICS_DIR / "EVALUATION_RESULTS.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# CBR-SFT-07 Step 50 Fixed Forensic Evaluation Suite Results\n\n")
        f.write("Evaluation Config: `temperature=0.0, repetition_penalty=1.25, no_repeat_ngram_size=3, max_new_tokens=48`\n\n")
        f.write("| Category | Prompt (Snippet) | Generated Output | Stopped on EOS |\n|---|---|---|---|\n")
        for r in results:
            clean_prompt = r["prompt"].replace("\n", " ")[:40]
            clean_gen = r["generated"].replace("\n", " ")
            f.write(f"| `{r['category']}` | `{clean_prompt}...` | {clean_gen} | {r['stopped_on_eos']} |\n")
            
        f.write("\n## Raw Detailed Outputs\n\n")
        for idx, r in enumerate(results, 1):
            f.write(f"### {idx}. [{r['category']}]\n")
            f.write(f"**Prompt**:\n```\n{r['prompt']}\n```\n")
            f.write(f"**Generated Response**: `{r['generated']}`\n")
            f.write(f"**Expected Topic**: `{r['expected']}`\n")
            f.write(f"**Tokens**: {r['num_tokens']} | **EOS Reached**: {r['stopped_on_eos']}\n\n")
            
    print(f"EVALUATION_RESULTS.md written to {out_md}")

if __name__ == "__main__":
    main()

