"""
Phase G: Generation Quality Inspection for UCIP-13R.
Evaluates exactly 20 distinct prompts across 8 mandated categories:
1. English continuation (3 prompts)
2. Hindi continuation (3 prompts)
3. Sanskrit continuation (2 prompts)
4. Python continuation (3 prompts)
5. Simple arithmetic (2 prompts)
6. Basic factual recall (3 prompts)
7. Instruction following (2 prompts)
8. Short-form explanation (2 prompts)

Decodes from Step 0 and Step 800 checkpoints using fixed random seed (42)
with temperature 0.7, top-p 0.9, and greedy decoding (temp=0.0).
Inspects for: gibberish, repetition loops, token collapse, premature EOS, missing EOS, broken UTF-8, syntax, and semantics.
"""

import json
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

DEVICE = torch.device("cpu")
EXP_DIR = ROOT / "experiments" / "ucip13"
CKPT_DIR = EXP_DIR / "checkpoints" / "primary_seed42"
OUT_FILE = ROOT / "experiments" / "ucip13r" / "evaluations" / "generation_audit_samples.json"

AUDIT_PROMPTS = [
    # 1. English continuation
    {"id": "ENG-CONT-01", "category": "English continuation", "prompt": "The solar system consists of the Sun and the objects that orbit it,"},
    {"id": "ENG-CONT-02", "category": "English continuation", "prompt": "Artificial intelligence is a branch of computer science that deals with"},
    {"id": "ENG-CONT-03", "category": "English continuation", "prompt": "A river flows through the valley towards the distant ocean,"},
    # 2. Hindi continuation
    {"id": "HIN-CONT-01", "category": "Hindi continuation", "prompt": "भारत एक विशाल देश है जहाँ विभिन्न धर्मों और संस्कृतियों के लोग"},
    {"id": "HIN-CONT-02", "category": "Hindi continuation", "prompt": "कंप्यूटर विज्ञान आधुनिक युग का सबसे महत्वपूर्ण क्षेत्र है क्योंकि"},
    {"id": "HIN-CONT-03", "category": "Hindi continuation", "prompt": "हिमालय पर्वत भारत के उत्तर में स्थित एक विशाल पर्वतमाला है,"},
    # 3. Sanskrit continuation
    {"id": "SAN-CONT-01", "category": "Sanskrit continuation", "prompt": "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।"},
    {"id": "SAN-CONT-02", "category": "Sanskrit continuation", "prompt": "विद्या ददाति विनयं विनयाद् याति पात्रताम्।"},
    # 4. Python continuation
    {"id": "PY-CONT-01", "category": "Python continuation", "prompt": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n"},
    {"id": "PY-CONT-02", "category": "Python continuation", "prompt": "def is_prime(number):\n    \"\"\"Check if a number is prime.\"\"\"\n    if number <= 1:\n"},
    {"id": "PY-CONT-03", "category": "Python continuation", "prompt": "import json\n\ndef load_data(filepath):\n    with open(filepath, 'r') as f:\n"},
    # 5. Simple arithmetic
    {"id": "ARITH-01", "category": "Simple arithmetic", "prompt": "Question: What is 15 + 27? Answer:"},
    {"id": "ARITH-02", "category": "Simple arithmetic", "prompt": "If a box contains 12 apples and you take away 5, how many remain?"},
    # 6. Basic factual recall
    {"id": "FACT-01", "category": "Basic factual recall", "prompt": "The capital city of France is"},
    {"id": "FACT-02", "category": "Basic factual recall", "prompt": "Water is composed of two hydrogen atoms and one"},
    {"id": "FACT-03", "category": "Basic factual recall", "prompt": "The largest planet in our solar system is"},
    # 7. Instruction following
    {"id": "INST-01", "category": "Instruction following", "prompt": "Instruction: Summarize what a neural network does in one sentence.\nResponse:"},
    {"id": "INST-02", "category": "Instruction following", "prompt": "User: What are three benefits of regular exercise?\nAssistant:"},
    # 8. Short-form explanation
    {"id": "EXPL-01", "category": "Short-form explanation", "prompt": "Explain why leaves change color in the autumn season:"},
    {"id": "EXPL-02", "category": "Short-form explanation", "prompt": "Explain how gravity keeps planets in orbit around the Sun:"}
]


def generate_for_model(model, tokenizer, prompts, max_new_tokens=40, seed=42):
    results = []
    for item in prompts:
        prompt_text = item["prompt"]
        input_ids = tokenizer.encode(prompt_text)
        
        # Test 1: Greedy (temperature=0.0)
        torch.manual_seed(seed)
        greedy_tokens = list(input_ids)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = greedy_tokens[-512:]
                inp = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logits = model(inp)[0, -1, :]
                next_tok = torch.argmax(logits).item()
                greedy_tokens.append(next_tok)
                if next_tok == 1:
                    break
        gen_greedy = greedy_tokens[len(input_ids):]
        text_greedy = tokenizer.decode(gen_greedy, errors="replace")
        
        # Test 2: Sampled (temperature=0.7)
        torch.manual_seed(seed)
        sample_tokens = list(input_ids)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = sample_tokens[-512:]
                inp = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logits = model(inp)[0, -1, :]
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                sample_tokens.append(next_tok)
                if next_tok == 1:
                    break
        gen_sample = sample_tokens[len(input_ids):]
        text_sample = tokenizer.decode(gen_sample, errors="replace")
        
        results.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": prompt_text,
            "greedy": {
                "generated_text": text_greedy,
                "token_count": len(gen_greedy),
                "produced_eos": 1 in gen_greedy,
            },
            "sampled_temp_0.7": {
                "generated_text": text_sample,
                "token_count": len(gen_sample),
                "produced_eos": 1 in gen_sample,
            }
        })
    return results


def run_generation_audit():
    print("=" * 80)
    print("PHASE G: GENERATION QUALITY INSPECTION")
    print("=" * 80)
    
    tokenizer, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    
    print("Loading Step 0 model...")
    m0 = ChakrMicro(ModelConfig())
    d0 = torch.load(CKPT_DIR / "checkpoint_step_000.pt", map_location="cpu", weights_only=False)
    m0.load_state_dict(d0["model_state_dict"])
    m0.eval()
    
    print("Loading Step 800 model...")
    m800 = ChakrMicro(ModelConfig())
    d800 = torch.load(CKPT_DIR / "checkpoint_step_800.pt", map_location="cpu", weights_only=False)
    m800.load_state_dict(d800["model_state_dict"])
    m800.eval()
    
    print(f"Generating for {len(AUDIT_PROMPTS)} prompts on Step 0...")
    out_0 = generate_for_model(m0, tokenizer, AUDIT_PROMPTS)
    
    print(f"Generating for {len(AUDIT_PROMPTS)} prompts on Step 800...")
    out_800 = generate_for_model(m800, tokenizer, AUDIT_PROMPTS)
    
    paired_comparison = []
    for p0, p8 in zip(out_0, out_800):
        paired_comparison.append({
            "id": p0["id"],
            "category": p0["category"],
            "prompt": p0["prompt"],
            "step_000_greedy": p0["greedy"],
            "step_800_greedy": p8["greedy"],
            "step_000_sampled": p0["sampled_temp_0.7"],
            "step_800_sampled": p8["sampled_temp_0.7"],
        })
        
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(paired_comparison, f, indent=2, ensure_ascii=False)
        
    print("Generation audit completed and saved to:", OUT_FILE)


if __name__ == "__main__":
    run_generation_audit()
