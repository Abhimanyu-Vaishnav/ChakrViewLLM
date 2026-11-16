"""
generate_conversations.py - Generates 100 comparable conversations across:
- Parent Baseline (cbr_sft_04)
- v0.10 Candidate Seed 2026
- v0.10 Candidate Seed 3030

Categories (20 prompts each):
1. English
2. Devanagari Hindi
3. Roman Hindi
4. Hinglish
5. Multi-turn Context
"""

import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

OUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_20r" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    # --- 1. ENGLISH (20 prompts) ---
    {"id": "EN_01", "category": "english", "prompt": "User: Hello! Who are you?\nAssistant:"},
    {"id": "EN_02", "category": "english", "prompt": "User: Good morning! How can you help me today?\nAssistant:"},
    {"id": "EN_03", "category": "english", "prompt": "User: Can you explain what an algorithm is in simple terms?\nAssistant:"},
    {"id": "EN_04", "category": "english", "prompt": "User: What is the purpose of unit testing in software?\nAssistant:"},
    {"id": "EN_05", "category": "english", "prompt": "User: Could you give me an example of recursion?\nAssistant:"},
    {"id": "EN_06", "category": "english", "prompt": "User: How does a compiler translate code?\nAssistant:"},
    {"id": "EN_07", "category": "english", "prompt": "User: I feel overwhelmed with my project. Any advice?\nAssistant:"},
    {"id": "EN_08", "category": "english", "prompt": "User: What makes clean code important for teams?\nAssistant:"},
    {"id": "EN_09", "category": "english", "prompt": "User: Can you explain the difference between RAM and storage?\nAssistant:"},
    {"id": "EN_10", "category": "english", "prompt": "User: Tell me a brief story about an autonomous robot.\nAssistant:"},
    {"id": "EN_11", "category": "english", "prompt": "User: What is an API and why do developers use it?\nAssistant:"},
    {"id": "EN_12", "category": "english", "prompt": "User: What are the main features of Python?\nAssistant:"},
    {"id": "EN_13", "category": "english", "prompt": "User: How do databases index data for faster queries?\nAssistant:"},
    {"id": "EN_14", "category": "english", "prompt": "User: Are you capable of learning from our conversation?\nAssistant:"},
    {"id": "EN_15", "category": "english", "prompt": "User: Good evening! Are you online right now?\nAssistant:"},
    {"id": "EN_16", "category": "english", "prompt": "User: What is open source software?\nAssistant:"},
    {"id": "EN_17", "category": "english", "prompt": "User: Explain how version control works.\nAssistant:"},
    {"id": "EN_18", "category": "english", "prompt": "User: How do neural networks process information?\nAssistant:"},
    {"id": "EN_19", "category": "english", "prompt": "User: Thank you for your assistance today.\nAssistant:"},
    {"id": "EN_20", "category": "english", "prompt": "User: Goodbye! Have a great evening.\nAssistant:"},

    # --- 2. DEVANAGARI HINDI (20 prompts) ---
    {"id": "HI_01", "category": "hindi_devanagari", "prompt": "User: नमस्ते! आप कौन हैं?\nAssistant:"},
    {"id": "HI_02", "category": "hindi_devanagari", "prompt": "User: शुभ प्रभात! आप कैसे हैं?\nAssistant:"},
    {"id": "HI_03", "category": "hindi_devanagari", "prompt": "User: आप क्या-क्या काम कर सकते हैं?\nAssistant:"},
    {"id": "HI_04", "category": "hindi_devanagari", "prompt": "User: कंप्यूटर प्रोग्रामिंग क्या है? सरल शब्दों में बताएं।\nAssistant:"},
    {"id": "HI_05", "category": "hindi_devanagari", "prompt": "User: मशीन लर्निंग का मुख्य उद्देश्य क्या होता है?\nAssistant:"},
    {"id": "HI_06", "category": "hindi_devanagari", "prompt": "User: पायथन भाषा क्यों इतनी लोकप्रिय है?\nAssistant:"},
    {"id": "HI_07", "category": "hindi_devanagari", "prompt": "User: सॉफ्टवेयर टेस्टिंग क्यों जरूरी है?\nAssistant:"},
    {"id": "HI_08", "category": "hindi_devanagari", "prompt": "User: एल्गोरिदम क्या होता है?\nAssistant:"},
    {"id": "HI_09", "category": "hindi_devanagari", "prompt": "User: आज का दिन कैसा है?\nAssistant:"},
    {"id": "HI_10", "category": "hindi_devanagari", "prompt": "User: क्या आप हिंदी समझ सकते हैं?\nAssistant:"},
    {"id": "HI_11", "category": "hindi_devanagari", "prompt": "User: इंटरनेट कैसे काम करता है?\nAssistant:"},
    {"id": "HI_12", "category": "hindi_devanagari", "prompt": "User: डेटाबेस क्या होता है?\nAssistant:"},
    {"id": "HI_13", "category": "hindi_devanagari", "prompt": "User: क्लाउड कंप्यूटिंग के क्या फायदे हैं?\nAssistant:"},
    {"id": "HI_14", "category": "hindi_devanagari", "prompt": "User: साइबर सुरक्षा क्यों महत्वपूर्ण है?\nAssistant:"},
    {"id": "HI_15", "category": "hindi_devanagari", "prompt": "User: शुभ संध्या! क्या आप तैयार हैं?\nAssistant:"},
    {"id": "HI_16", "category": "hindi_devanagari", "prompt": "User: कोडिंग सीखने की शुरुआत कैसे करें?\nAssistant:"},
    {"id": "HI_17", "category": "hindi_devanagari", "prompt": "User: आपका नाम क्या है?\nAssistant:"},
    {"id": "HI_18", "category": "hindi_devanagari", "prompt": "User: क्या आप मेरी पढ़ाई में मदद कर सकते हैं?\nAssistant:"},
    {"id": "HI_19", "category": "hindi_devanagari", "prompt": "User: बहुत-बहुत धन्यवाद!\nAssistant:"},
    {"id": "HI_20", "category": "hindi_devanagari", "prompt": "User: अलविदा! फिर मिलेंगे।\nAssistant:"},

    # --- 3. ROMAN HINDI (20 prompts) ---
    {"id": "ROM_01", "category": "roman_hindi", "prompt": "User: Tumhara naam kya hai?\nAssistant:"},
    {"id": "ROM_02", "category": "roman_hindi", "prompt": "User: Kaise ho bhai? Sab badhiya?\nAssistant:"},
    {"id": "ROM_03", "category": "roman_hindi", "prompt": "User: Mujhe samjhao ye AI kaise kaam karta hai?\nAssistant:"},
    {"id": "ROM_04", "category": "roman_hindi", "prompt": "User: Kya kar rahe ho abhi?\nAssistant:"},
    {"id": "ROM_05", "category": "roman_hindi", "prompt": "User: Ye program kaise run karein?\nAssistant:"},
    {"id": "ROM_06", "category": "roman_hindi", "prompt": "User: Kal maine kya bataya tha yaad hai?\nAssistant:"},
    {"id": "ROM_07", "category": "roman_hindi", "prompt": "User: Python sikhna mushkil hai kya?\nAssistant:"},
    {"id": "ROM_08", "category": "roman_hindi", "prompt": "User: Debugging karne ka sabse badhiya tareeka batao.\nAssistant:"},
    {"id": "ROM_09", "category": "roman_hindi", "prompt": "User: Bhai mujhe ek simple code samjha do.\nAssistant:"},
    {"id": "ROM_10", "category": "roman_hindi", "prompt": "User: Database mein index kya hota hai?\nAssistant:"},
    {"id": "ROM_11", "category": "roman_hindi", "prompt": "User: Git commit aur push mein kya farak hai?\nAssistant:"},
    {"id": "ROM_12", "category": "roman_hindi", "prompt": "User: Aaj ka din kaisa chal raha hai?\nAssistant:"},
    {"id": "ROM_13", "category": "roman_hindi", "prompt": "User: Kya tum meri help kar sakte ho coding mein?\nAssistant:"},
    {"id": "ROM_14", "category": "roman_hindi", "prompt": "User: Machine learning ke baare mein kuch batao.\nAssistant:"},
    {"id": "ROM_15", "category": "roman_hindi", "prompt": "User: Shukriya bhai help ke liye.\nAssistant:"},
    {"id": "ROM_16", "category": "roman_hindi", "prompt": "User: Acha suno, ek sawaal tha.\nAssistant:"},
    {"id": "ROM_17", "category": "roman_hindi", "prompt": "User: Ye function kya return karega?\nAssistant:"},
    {"id": "ROM_18", "category": "roman_hindi", "prompt": "User: Frontend aur backend mein kya antar hai?\nAssistant:"},
    {"id": "ROM_19", "category": "roman_hindi", "prompt": "User: Main naya programmer hoon, kya advice doge?\nAssistant:"},
    {"id": "ROM_20", "category": "roman_hindi", "prompt": "User: Chalo theek hai, baad mein baat karte hain.\nAssistant:"},

    # --- 4. HINGLISH (20 prompts) ---
    {"id": "HING_01", "category": "hinglish", "prompt": "User: Bhai, mujhe Python samjhao.\nAssistant:"},
    {"id": "HING_02", "category": "hinglish", "prompt": "User: What is machine learning? Simple Hindi mein samjhao.\nAssistant:"},
    {"id": "HING_03", "category": "hinglish", "prompt": "User: Explain REST API in simple Hinglish.\nAssistant:"},
    {"id": "HING_04", "category": "hinglish", "prompt": "User: Code review karte waqt kya-kya dhyan rakhna chahiye?\nAssistant:"},
    {"id": "HING_05", "category": "hinglish", "prompt": "User: Docker container kya hota hai aur iska use kya hai?\nAssistant:"},
    {"id": "HING_06", "category": "hinglish", "prompt": "User: What are microservices? Hinglish mein batao.\nAssistant:"},
    {"id": "HING_07", "category": "hinglish", "prompt": "User: Mere code mein bug aa raha hai, kaise debug karoon?\nAssistant:"},
    {"id": "HING_08", "category": "hinglish", "prompt": "User: SQL aur NoSQL databases mein main differences kya hain?\nAssistant:"},
    {"id": "HING_09", "category": "hinglish", "prompt": "User: Clean code principles follow karna kyun important hai?\nAssistant:"},
    {"id": "HING_10", "category": "hinglish", "prompt": "User: What is asynchronous programming? Simple language mein samjhao.\nAssistant:"},
    {"id": "HING_11", "category": "hinglish", "prompt": "User: Bhai GitHub par pull request kaise create karte hain?\nAssistant:"},
    {"id": "HING_12", "category": "hinglish", "prompt": "User: What is caching in web development?\nAssistant:"},
    {"id": "HING_13", "category": "hinglish", "prompt": "User: Recursion ka real world example do na.\nAssistant:"},
    {"id": "HING_14", "category": "hinglish", "prompt": "User: Object Oriented Programming ke four pillars kaun se hain?\nAssistant:"},
    {"id": "HING_15", "category": "hinglish", "prompt": "User: Bhai model ki accuracy kaise improve karein?\nAssistant:"},
    {"id": "HING_16", "category": "hinglish", "prompt": "User: CI/CD pipeline ka concept explain karo.\nAssistant:"},
    {"id": "HING_17", "category": "hinglish", "prompt": "User: Cloud deployment ke liye AWS better hai ya GCP?\nAssistant:"},
    {"id": "HING_18", "category": "hinglish", "prompt": "User: Fastapi framework ke kya advantages hain over Flask?\nAssistant:"},
    {"id": "HING_19", "category": "hinglish", "prompt": "User: Great explanation bhai, thanks a lot!\nAssistant:"},
    {"id": "HING_20", "category": "hinglish", "prompt": "User: See you later, take care!\nAssistant:"},

    # --- 5. MULTI-TURN CONTEXT (20 dialogues) ---
    {
        "id": "CTX_01", "category": "multiturn_context",
        "prompt": "User: My name is Aarav. I live in Delhi.\nAssistant: Nice to meet you, Aarav from Delhi!\nUser: What is my name?\nAssistant:"
    },
    {
        "id": "CTX_02", "category": "multiturn_context",
        "prompt": "User: Mera naam Priya hai aur main Mumbai se hoon.\nAssistant: Hello Priya! Accha laga jaan kar ki aap Mumbai se ho.\nUser: Mera naam kya hai?\nAssistant:"
    },
    {
        "id": "CTX_03", "category": "multiturn_context",
        "prompt": "User: My project is named Velora.\nAssistant: Understood, your project is Velora.\nUser: What is the name of my project?\nAssistant:"
    },
    {
        "id": "CTX_04", "category": "multiturn_context",
        "prompt": "User: I am building a neural network in Rust.\nAssistant: Building a neural network in Rust sounds exciting!\nUser: Can you recommend a compiler?\nAssistant: The official rustc compiler and cargo package manager are the standard.\nUser: Which language did I say I am using?\nAssistant:"
    },
    {
        "id": "CTX_05", "category": "multiturn_context",
        "prompt": "User: I am in Jaipur today.\nAssistant: Jaipur is a wonderful city.\nUser: Actually, I moved to Bengaluru.\nAssistant: Noted your update to Bengaluru.\nUser: Which city am I currently in?\nAssistant:"
    },
    {
        "id": "CTX_06", "category": "multiturn_context",
        "prompt": "User: Mera favorite color blue hai.\nAssistant: Blue ek bahut sundar color hai.\nUser: Mera favorite color kya hai?\nAssistant:"
    },
    {
        "id": "CTX_07", "category": "multiturn_context",
        "prompt": "User: Let us talk about Python.\nAssistant: Sure, what about Python interests you?\nUser: Can you explain lists?\nAssistant: Lists in Python are mutable ordered sequences.\nUser: What language were we discussing?\nAssistant:"
    },
    {
        "id": "CTX_08", "category": "multiturn_context",
        "prompt": "User: My server IP is 192.168.1.100.\nAssistant: Server IP recorded.\nUser: What is my server IP?\nAssistant:"
    },
    {
        "id": "CTX_09", "category": "multiturn_context",
        "prompt": "User: Main Indore se hoon.\nAssistant: Indore ek saaf aur khubsurat shahar hai.\nUser: Main kahan se hoon?\nAssistant:"
    },
    {
        "id": "CTX_10", "category": "multiturn_context",
        "prompt": "User: My primary framework is PyTorch.\nAssistant: PyTorch is excellent for deep learning research.\nUser: How is the weather?\nAssistant: I do not have real-time weather information.\nUser: Which framework did I mention earlier?\nAssistant:"
    },
    {
        "id": "CTX_11", "category": "multiturn_context",
        "prompt": "User: The database port is 5432.\nAssistant: Port 5432 noted.\nUser: Which port is the database using?\nAssistant:"
    },
    {
        "id": "CTX_12", "category": "multiturn_context",
        "prompt": "User: Mera naam Rohan hai.\nAssistant: Hello Rohan!\nUser: Kal meeting hai 10 baje.\nAssistant: Meeting at 10 AM noted.\nUser: Mera naam kya hai?\nAssistant:"
    },
    {
        "id": "CTX_13", "category": "multiturn_context",
        "prompt": "User: I have 4 GPUs on my cluster.\nAssistant: That is a solid compute setup.\nUser: How many GPUs do I have?\nAssistant:"
    },
    {
        "id": "CTX_14", "category": "multiturn_context",
        "prompt": "User: Main React use kar raha hoon frontend ke liye.\nAssistant: React frontend ke liye standard choice hai.\nUser: Frontend ke liye main kya use kar raha hoon?\nAssistant:"
    },
    {
        "id": "CTX_15", "category": "multiturn_context",
        "prompt": "User: Set parameter alpha to 0.05.\nAssistant: Parameter alpha set to 0.05.\nUser: What is the value of alpha?\nAssistant:"
    },
    {
        "id": "CTX_16", "category": "multiturn_context",
        "prompt": "User: Our product is called ChakrView.\nAssistant: ChakrView is an inspiring project.\nUser: What is our product called?\nAssistant:"
    },
    {
        "id": "CTX_17", "category": "multiturn_context",
        "prompt": "User: Mera hometown Bhopal hai.\nAssistant: Bhopal jheelo ka shahar hai.\nUser: Mera hometown kaun sa hai?\nAssistant:"
    },
    {
        "id": "CTX_18", "category": "multiturn_context",
        "prompt": "User: The training batch size is 64.\nAssistant: Batch size 64 recorded.\nUser: What batch size are we using?\nAssistant:"
    },
    {
        "id": "CTX_19", "category": "multiturn_context",
        "prompt": "User: My username is admin_dev.\nAssistant: Username acknowledged.\nUser: What is my username?\nAssistant:"
    },
    {
        "id": "CTX_20", "category": "multiturn_context",
        "prompt": "User: Main Hindi seekhna chahta hoon.\nAssistant: Hindi seekhna bahut accha decision hai.\nUser: Main kya seekhna chahta hoon?\nAssistant:"
    }
]

def generate_responses_for_model(model_name: str, ckpt_path: Path):
    print(f"\n=======================================================")
    print(f"  GENERATING CONVERSATIONS: {model_name}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"=======================================================")
    runner = DirectNeuralRunner(ckpt_path)
    tok = runner.tokenizer
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)

    records = []
    t0 = time.time()
    for item in PROMPTS:
        pid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

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
        has_eos = (1 in gen)
        records.append({
            "conversation_id": pid,
            "category": cat,
            "model_name": model_name,
            "checkpoint_name": ckpt_path.name,
            "prompt": prompt,
            "response": resp,
            "input_tokens": len(toks),
            "output_tokens": len(gen),
            "has_eos": has_eos
        })

    elapsed = time.time() - t0
    print(f"Completed 100 conversations for {model_name} in {elapsed:.1f}s")
    return records

def main():
    ckpts = [
        ("parent_baseline", REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"),
        ("v010_seed2026", REPO_ROOT / "releases/v0.10_conversational_model/chakrview_v010_model.pt"),
        ("v010_seed3030", REPO_ROOT / "experiments/cognition/cr_exp_20/checkpoints/cr20_candidate_seed3030_best.pt")
    ]

    all_conversations = []
    for name, p in ckpts:
        records = generate_responses_for_model(name, p)
        all_conversations.extend(records)

    out_file = OUT_DIR / "raw_conversations.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_conversations:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_conversations)} conversation traces to {out_file}")

if __name__ == "__main__":
    main()

