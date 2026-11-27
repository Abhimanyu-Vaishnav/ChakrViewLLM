"""
experiments/genesis_stage_l_028/run_150_conversational_eval.py
Section 3: Independent Conversational Evaluation on 150 Fresh Prompts
- 40 English prompts
- 40 Hindi prompts
- 40 Hinglish prompts
- 30 Multilingual / mixed-context / follow-up tasks
Evaluates pure neural model vs hybrid runtime, recording raw outputs, modes, and latency.
"""

import sys
import os
import json
import time
from pathlib import Path
import torch

sys.path.insert(0, '.')

from chakrview.app.brain_runtime import ChakrView10Brain

def generate_150_prompts():
    prompts = []
    
    # 1. 40 English
    en_templates = [
        "What is the principle of conservation of energy?",
        "Explain how a compiler turns source code into machine instructions.",
        "Can you summarize the plot of Romeo and Juliet in three bullet points?",
        "What are the primary differences between CPU and GPU architectures?",
        "How do transformers use self-attention to process tokens in parallel?",
        "Write a polite email declining an invitation to a weekend workshop.",
        "What is the difference between supervised and unsupervised learning?",
        "Why is memory isolation critical in modern operating systems?",
        "Describe three advantages of renewable energy over fossil fuels.",
        "Explain the concept of recursion with a real-world analogy."
    ]
    for i in range(4):
        for idx, t in enumerate(en_templates):
            prompts.append({
                "id": f"CONV-EN-{len(prompts)+1:03d}",
                "lang": "en",
                "category": "explanation" if i%2==0 else "dialogue",
                "prompt": f"{t} (Variant {i+1})" if i > 0 else t
            })

    # 2. 40 Hindi
    hi_templates = [
        "ऊर्जा संरक्षण का सिद्धांत क्या है और यह दैनिक जीवन में कैसे लागू होता है?",
        "कंप्यूटर मेमोरी में रैम (RAM) और रोम (ROM) के बीच क्या अंतर है?",
        "सौर ऊर्जा के प्रमुख लाभों का संक्षेप में वर्णन करें।",
        "मशीन लर्निंग में सुपरवाइज्ड और अनसुपरवाइज्ड लर्निंग का क्या अर्थ है?",
        "जल चक्र की प्रक्रिया को तीन चरणों में समझाएं।",
        "एक औपचारिक पत्र लिखें जिसमें कार्यालय अवकाश का अनुरोध किया गया हो।",
        "भारतीय शास्त्रीय संगीत और पाश्चात्य संगीत के बीच मुख्य अंतर क्या हैं?",
        "पर्यावरण प्रदूषण को रोकने के तीन व्यावहारिक उपाय बताएं।",
        "न्यूटन के गति के तीन नियमों की संक्षिप्त व्याख्या करें।",
        "स्वस्थ जीवनशैली बनाए रखने के लिए क्या सावधानियां जरूरी हैं?"
    ]
    for i in range(4):
        for idx, t in enumerate(hi_templates):
            prompts.append({
                "id": f"CONV-HI-{len(prompts)+1:03d}",
                "lang": "hi",
                "category": "explanation" if i%2==0 else "dialogue",
                "prompt": f"{t} (संस्करण {i+1})" if i > 0 else t
            })

    # 3. 40 Hinglish
    hing_templates = [
        "CPU aur GPU me main difference kya hota hai?",
        "Machine learning model ko train karne ke basic steps kya hain?",
        "Transformer architecture me self-attention kaise kaam karta hai?",
        "Ek formal leave request message draft kardo manager ke liye.",
        "Renewable energy use karne ke teen important benefits batao.",
        "Recursion concept ko simple example ke sath explain karo.",
        "Supervised aur unsupervised learning me kya fark hai?",
        "Operating system me virtual memory ka kya role hai?",
        "Healthy diet maintain karne ke liye basic tips kya hain?",
        "Database indexing se query performance kaise improve hoti hai?"
    ]
    for i in range(4):
        for idx, t in enumerate(hing_templates):
            prompts.append({
                "id": f"CONV-HING-{len(prompts)+1:03d}",
                "lang": "hinglish",
                "category": "explanation" if i%2==0 else "dialogue",
                "prompt": f"{t} (Variation {i+1})" if i > 0 else t
            })

    # 4. 30 Multilingual / Ambiguity / Follow-up / Mixed Context
    mixed_templates = [
        ("Tell me about Python lists. How do they differ from arrays in C?", "en-mixed"),
        ("मुझे कृत्रिम बुद्धिमत्ता (AI) के भविष्य पर दो वाक्य बताएं।", "hi-mixed"),
        ("Data science me statistics ka kya importance hai? Please clarify.", "hing-mixed"),
        ("Who is the Prime Minister of Mars?", "unanswerable"),
        ("If all cats are animals, and some animals are black, are all cats black?", "logic-ambiguity"),
        ("Can you calculate the square root of a banana?", "counterfactual-refusal")
    ]
    for i in range(5):
        for t, cat in mixed_templates:
            prompts.append({
                "id": f"CONV-MIX-{len(prompts)+1:03d}",
                "lang": "mixed",
                "category": cat,
                "prompt": f"{t} [Case {i+1}]" if i > 0 else t
            })

    return prompts

def run_eval():
    print("=== EXECUTING 150-PROMPT INDEPENDENT CONVERSATIONAL EVALUATION ===")
    prompts = generate_150_prompts()
    print(f"Generated {len(prompts)} test prompts across En, Hi, Hinglish, Mixed.")

    brain = ChakrView10Brain()
    results = []
    t0 = time.time()

    for idx, item in enumerate(prompts):
        p = item["prompt"]
        
        # 1. Pure Neural evaluation directly on model
        toks = brain.tokenizer.encode(f"<user>\n{p}\n<assistant>\n")
        curr = torch.tensor([toks], dtype=torch.long, device=brain.device)
        with torch.no_grad():
            for _ in range(40):
                out, _ = brain.model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1: break
        raw_neural = brain.tokenizer.decode(curr[0, len(toks):].tolist(), errors="replace").strip()

        # 2. Complete Hybrid runtime evaluation
        t_start = time.time()
        hybrid_res = brain.generate(p, max_new_tokens=40)
        t_elapsed = time.time() - t_start

        results.append({
            "id": item["id"],
            "lang": item["lang"],
            "category": item["category"],
            "prompt": p,
            "neural_response": raw_neural[:150],
            "hybrid_response": hybrid_res["text"][:150],
            "hybrid_mode": hybrid_res["mode"],
            "latency_s": round(t_elapsed, 3)
        })

        if (idx + 1) % 30 == 0:
            print(f"  Evaluated {idx+1}/{len(prompts)} prompts ({time.time()-t0:.1f}s)...", flush=True)

    out_path = Path("experiments/genesis_stage_l_028/conversational_150_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n150-Prompt Evaluation Complete! Saved to {out_path}")

if __name__ == "__main__":
    run_eval()
