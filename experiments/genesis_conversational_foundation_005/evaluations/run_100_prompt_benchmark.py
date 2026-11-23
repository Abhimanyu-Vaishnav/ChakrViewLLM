"""
Workstream C: Comprehensive Independent Evaluation System for Mission CV-GENESIS-NR-005.

Evaluates 3 model checkpoints:
1. Canonical Pretrained Best (experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt)
2. SFT Best Checkpoint (experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt)
3. SFT Final Checkpoint (experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt)

Benchmark covers 100 held-out prompts across:
- English Conversation & Explanations (20)
- Devanagari Hindi Natural Dialogue & QA (20)
- Hinglish Conversations & Practical QA (20)
- Technical Concepts & Python Code (15)
- Basic Arithmetic & Algorithmic Reasoning (10)
- Context-Grounding & Multi-turn Consistency (10)
- Principled Abstention on Unknowns / Future (5)

Pure Neural isolation: zero external APIs, zero templates, zero hybrid interceptors.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"

EVALUATION_100_PROMPTS = [
    # English Conversation & Explanations (20)
    ("en", "conv", "User: Hi there! How are you doing today?\nAssistant:"),
    ("en", "conv", "User: Good morning, what can you help me with?\nAssistant:"),
    ("en", "conv", "User: Who are you and what is your purpose?\nAssistant:"),
    ("en", "conv", "User: Can you tell me a little bit about yourself?\nAssistant:"),
    ("en", "conv", "User: I had a very busy day at work.\nAssistant:"),
    ("en", "expl", "User: Why does a computer need RAM in addition to storage?\nAssistant:"),
    ("en", "expl", "User: Explain the difference between compile-time and runtime.\nAssistant:"),
    ("en", "expl", "User: What is an operating system and why is it essential?\nAssistant:"),
    ("en", "expl", "User: How does the internet transfer data across continents?\nAssistant:"),
    ("en", "expl", "User: What causes the seasons to change on Earth?\nAssistant:"),
    ("en", "expl", "User: What is photosynthesis in simple terms?\nAssistant:"),
    ("en", "expl", "User: How does encryption protect confidential messages?\nAssistant:"),
    ("en", "expl", "User: Describe how an electric motor converts electrical energy into motion.\nAssistant:"),
    ("en", "expl", "User: What is the greenhouse effect and why does it matter?\nAssistant:"),
    ("en", "expl", "User: What is the role of red blood cells in the human body?\nAssistant:"),
    ("en", "fact", "User: What is the capital of Japan?\nAssistant:"),
    ("en", "fact", "User: What is the largest planet in our solar system?\nAssistant:"),
    ("en", "fact", "User: In which continent is Egypt located?\nAssistant:"),
    ("en", "fact", "User: What chemical element has the symbol O?\nAssistant:"),
    ("en", "fact", "User: Who wrote the play Romeo and Juliet?\nAssistant:"),

    # Hindi Conversation & QA (20)
    ("hi", "conv", "User: नमस्ते, आप कैसे हैं?\nAssistant:"),
    ("hi", "conv", "User: आपका नाम क्या है और आप क्या कर सकते हैं?\nAssistant:"),
    ("hi", "conv", "User: शुभ प्रभात! आज का दिन कैसा है?\nAssistant:"),
    ("hi", "conv", "User: मुझे आपसे मिलकर बहुत खुशी हुई।\nAssistant:"),
    ("hi", "conv", "User: क्या आप हिंदी में बातचीत कर सकते हैं?\nAssistant:"),
    ("hi", "expl", "User: कंप्यूटर में सीपीयू का क्या मुख्य कार्य होता है?\nAssistant:"),
    ("hi", "expl", "User: इंटरनेट क्या है और यह कैसे काम करता है?\nAssistant:"),
    ("hi", "expl", "User: पेड़-पौधे हमारे पर्यावरण के लिए क्यों आवश्यक हैं?\nAssistant:"),
    ("hi", "expl", "User: सौर ऊर्जा क्या होती है और इसके क्या लाभ हैं?\nAssistant:"),
    ("hi", "expl", "User: वर्षा ऋतु में बारिश कैसे होती है?\nAssistant:"),
    ("hi", "fact", "User: भारत की राजधानी क्या है?\nAssistant:"),
    ("hi", "fact", "User: हमारे सौरमंडल में कितने ग्रह हैं?\nAssistant:"),
    ("hi", "fact", "User: भारत का राष्ट्रीय पशु कौन सा है?\nAssistant:"),
    ("hi", "fact", "User: जल का रासायनिक सूत्र क्या है?\nAssistant:"),
    ("hi", "fact", "User: ताजमहल किस शहर में स्थित है?\nAssistant:"),
    ("hi", "arith", "User: 15 में 7 जोड़ने पर क्या प्राप्त होगा?\nAssistant:"),
    ("hi", "arith", "User: 40 को 5 से भाग देने पर क्या आएगा?\nAssistant:"),
    ("hi", "inst", "User: एक से पाँच तक गिनती लिखकर दिखाइए।\nAssistant:"),
    ("hi", "inst", "User: एक छोटी सी कविता या दो पंक्ति लिखिए।\nAssistant:"),
    ("hi", "conv", "User: धन्यवाद, आपकी सहायता बहुत अच्छी थी!\nAssistant:"),

    # Hinglish Conversation & QA (20)
    ("hinglish", "conv", "User: Bhai kaise ho, kya hal chal hai?\nAssistant:"),
    ("hinglish", "conv", "User: Tum kaun ho aur kya kaam karte ho?\nAssistant:"),
    ("hinglish", "conv", "User: Yaar mujhe coding seekhni hai, kahan se start karun?\nAssistant:"),
    ("hinglish", "conv", "User: Aaj mera exam ka result aane wala hai, thoda tension ho raha hai.\nAssistant:"),
    ("hinglish", "conv", "User: Koi badhiya si tech news batao.\nAssistant:"),
    ("hinglish", "expl", "User: RAM aur ROM me kya farak hota hai, aasan shabdon me samjhao.\nAssistant:"),
    ("hinglish", "expl", "User: AI model train karte time overfitting ka kya matlab hota hai?\nAssistant:"),
    ("hinglish", "expl", "User: Cloud computing kya hoti hai aur companies ise kyun use karti hain?\nAssistant:"),
    ("hinglish", "expl", "User: Machine learning me gradient descent algorithm kaise work karta hai?\nAssistant:"),
    ("hinglish", "expl", "User: Git me commit aur push me kya antar hai?\nAssistant:"),
    ("hinglish", "fact", "User: Bharat me sabse pehle metro train kis shahar me shuru hui thi?\nAssistant:"),
    ("hinglish", "fact", "User: Computer ka brain kis part ko kehte hain?\nAssistant:"),
    ("hinglish", "fact", "User: Python programming language kisne create ki thi?\nAssistant:"),
    ("hinglish", "arith", "User: 12 ko 6 se multiply karenge toh kitna aayega?\nAssistant:"),
    ("hinglish", "arith", "User: Agar 100 me se 35 minus karein toh kya bachega?\nAssistant:"),
    ("hinglish", "coding", "User: Python me ek number even hai ya odd, check karne ka code do.\nAssistant:"),
    ("hinglish", "coding", "User: Ek list ko reverse karne ka python function likho.\nAssistant:"),
    ("hinglish", "inst", "User: Mujhe 3 daily habits batao jo focus increase karein.\nAssistant:"),
    ("hinglish", "conv", "User: Sahi hai bhai, maza aa gaya baat karke.\nAssistant:"),
    ("hinglish", "conv", "User: Theek hai dost, fir milte hain!\nAssistant:"),

    # Technical Concepts & Coding (15)
    ("en", "coding", "User: Write a Python function to find the maximum element in a list of numbers.\nAssistant:"),
    ("en", "coding", "User: How do you read a text file line by line in Python safely?\nAssistant:"),
    ("en", "coding", "User: Write a function to check if a given string is a palindrome.\nAssistant:"),
    ("en", "coding", "User: Implement binary search on a sorted integer array in Python.\nAssistant:"),
    ("en", "coding", "User: Show an example of using a dictionary in Python.\nAssistant:"),
    ("en", "tech", "User: What is the time complexity of searching in a balanced binary search tree?\nAssistant:"),
    ("en", "tech", "User: Explain the difference between TCP and UDP protocols.\nAssistant:"),
    ("en", "tech", "User: What is a deadlock in concurrent multi-threaded programming?\nAssistant:"),
    ("en", "tech", "User: How does attention mechanism work in a Transformer model?\nAssistant:"),
    ("en", "tech", "User: What is the difference between supervised and unsupervised learning?\nAssistant:"),
    ("en", "tech", "User: Explain the concept of ACID properties in relational databases.\nAssistant:"),
    ("en", "tech", "User: What is the role of an activation function in a neural network layer?\nAssistant:"),
    ("en", "tech", "User: What is the difference between a stack and a queue data structure?\nAssistant:"),
    ("en", "tech", "User: How does garbage collection work in high-level programming languages?\nAssistant:"),
    ("en", "tech", "User: What is a REST API and what HTTP methods are commonly used?\nAssistant:"),

    # Arithmetic & Algorithmic Reasoning (10)
    ("en", "arith", "User: Calculate 24 + 37.\nAssistant:"),
    ("en", "arith", "User: What is 85 - 29?\nAssistant:"),
    ("en", "arith", "User: Compute 14 * 5.\nAssistant:"),
    ("en", "arith", "User: What is 144 divided by 12?\nAssistant:"),
    ("en", "arith", "User: If a train travels at 60 km/h for 3 hours, what total distance does it cover?\nAssistant:"),
    ("en", "reasoning", "User: I have 4 baskets and each basket holds 6 apples. How many apples in total?\nAssistant:"),
    ("en", "reasoning", "User: If today is Monday, what day of the week will it be in 3 days?\nAssistant:"),
    ("en", "reasoning", "User: Which number is larger: 0.75 or 0.8?\nAssistant:"),
    ("en", "reasoning", "User: A box contains 10 red pens and 5 blue pens. How many pens are in the box?\nAssistant:"),
    ("en", "reasoning", "User: If all squares are rectangles, is every square a rectangle?\nAssistant:"),

    # Context-Grounding & Consistency (10)
    ("en", "context", "User: Project Falcon is scheduled for release on October 15. When is Project Falcon scheduled?\nAssistant:"),
    ("en", "context", "User: The server IP address is 10.0.4.12. What is the server IP?\nAssistant:"),
    ("en", "context", "User: Dr. Sharma lives in Jaipur and works at the central hospital. Where does Dr. Sharma live?\nAssistant:"),
    ("en", "context", "User: Sarah bought 3 notebooks and 2 pencils. What did Sarah buy?\nAssistant:"),
    ("en", "context", "User: The key is hidden inside the blue drawer. Where is the key?\nAssistant:"),
    ("hinglish", "context", "User: Project Garuda ka internal port 8080 hai. Garuda ka port kya hai?\nAssistant:"),
    ("hinglish", "context", "User: Rohit Pune me rehta hai aur software engineer hai. Rohit kahan rehta hai?\nAssistant:"),
    ("hi", "context", "User: मोहन की दुकान जयपुर के जौहरी बाजार में है। मोहन की दुकान कहाँ है?\nAssistant:"),
    ("hi", "context", "User: रोहिणी ने परीक्षा में प्रथम स्थान प्राप्त किया। प्रथम स्थान किसने प्राप्त किया?\nAssistant:"),
    ("en", "context", "User: The secret access code is Delta-779. State the code.\nAssistant:"),

    # Principled Abstention (5)
    ("en", "abstain", "User: Who will win the cricket world cup in 2035?\nAssistant:"),
    ("en", "abstain", "User: What will be the exact stock price of Apple next Tuesday?\nAssistant:"),
    ("hi", "abstain", "User: क्या आप मुझे कल का मौसम बिना किसी डेटा के बिल्कुल सटीक बता सकते हैं?\nAssistant:"),
    ("hinglish", "abstain", "User: Kal lottery ka winning number kya aane wala hai?\nAssistant:"),
    ("en", "abstain", "User: What is the secret personal phone number of the Prime Minister?\nAssistant:")
]


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40, temperature: int = 0.7, top_k: int = 20, device: str = "cuda:0"):
    eos_id = 1
    p_ids = tokenizer.encode(prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -512:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_token = idx_next.item()
            if next_token == eos_id:
                break
            generated.append(next_token)
            idx = torch.cat((idx, idx_next), dim=1)
            
    try:
        return tokenizer.decode(generated, errors="replace").strip()
    except Exception:
        # Fallback raw byte decode with replacement
        raw_b = tokenizer.decode_bytes(generated)
        return raw_b.decode("utf-8", errors="replace").strip()


def run_benchmark(
    ckpts: dict[str, str],
    tokenizer_dir: str = "data/experiments/vocab_4096",
    output_dir: str = "experiments/genesis_conversational_foundation_005/evaluations",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tok, _ = load_tokenizer_artifacts(tokenizer_dir)
    results = {}
    
    for ckpt_name, ckpt_path in ckpts.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue
            
        print(f"\n=======================================================")
        print(f"EVALUATING MODEL: {ckpt_name} ({ckpt_path})")
        print(f"=======================================================")
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ChakrGenesis(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        model_evals = []
        for idx, (lang, cat, p) in enumerate(EVALUATION_100_PROMPTS):
            resp = generate_response(model, tok, p, max_new_tokens=40, temperature=0.7, top_k=20, device=device)
            model_evals.append({
                "index": idx + 1,
                "language": lang,
                "category": cat,
                "prompt": p,
                "response": resp
            })
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/100 prompts...")
                
        results[ckpt_name] = {
            "checkpoint_path": ckpt_path,
            "total_prompts": len(model_evals),
            "evaluations": model_evals
        }
        
    out_file = out_dir / "100_prompt_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull 100-prompt benchmark results saved to: {out_file}")


if __name__ == "__main__":
    checkpoints = {
        "canonical_pretrain_best": "experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt",
        "genesis_sft_best": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt",
        "genesis_sft_final": "experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_final.pt"
    }
    run_benchmark(checkpoints)
