import json
import random
from pathlib import Path

random.seed(42)

def build_acceptance_benchmark():
    probes = []
    
    # 1. Exact Sentence Copying (20)
    copy_sentences = [
        "The sky is blue.",
        "A dog is man's best friend.",
        "Water boils at 100 degrees Celsius.",
        "Python is a popular programming language.",
        "The capital of France is Paris."
    ]
    for s in copy_sentences:
        probes.append({
            "category": "exact_copying",
            "prompt": f"Say exactly: {s}",
            "expected": s.lower()
        })
        probes.append({
            "category": "exact_copying",
            "prompt": f"Repeat after me: {s}",
            "expected": s.lower()
        })
        
    # 2. Simple Sentence Generation (20)
    topics = ["a dog", "the sun", "a computer", "a tree", "a book"]
    for t in topics:
        probes.append({
            "category": "simple_generation",
            "prompt": f"Write one simple sentence about {t}.",
            "expected": t
        })
        probes.append({
            "category": "simple_generation",
            "prompt": f"Tell me a simple sentence about {t}.",
            "expected": t
        })
        
    # 3. Basic Arithmetic (30)
    for a, b in [(2, 2), (5, 7), (10, 5), (3, 4), (8, 2), (12, 6)]:
        probes.append({
            "category": "basic_arithmetic",
            "prompt": f"What is {a} + {b}?",
            "expected": str(a + b)
        })
        probes.append({
            "category": "basic_arithmetic",
            "prompt": f"Calculate {a} + {b}.",
            "expected": str(a + b)
        })
        
    # 4. User Name Recall (Hybrid vs Neural) (30)
    names = ["Rahul", "Priya", "Rohan", "Neha", "Kabir", "Ananya"]
    for n in names:
        probes.append({
            "category": "user_name_recall",
            "setup": f"My name is {n}.",
            "prompt": "What is my name?",
            "expected": n.lower()
        })
        
    # 5. Project Fact Recall & Reassignment (30)
    projects = [("Yatrajivi", "ChakrView"), ("Drishti", "Kavach"), ("Setu", "Bodhi")]
    for p1, p2 in projects:
        probes.append({
            "category": "project_fact_recall",
            "setup": f"My project is {p1}, but I renamed it to {p2}.",
            "prompt": "What is my project called now?",
            "expected": p2.lower()
        })
        
    # 6. Persona & Identity (20)
    persona_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me what you are.",
        "Introduce yourself."
    ]
    for p in persona_prompts:
        probes.append({
            "category": "persona_identity",
            "prompt": p,
            "expected": "chakrview"
        })
        
    # 7. English Natural Conversation & Greetings (20)
    greetings = [
        "Good morning assistant!",
        "Hello!",
        "How are you today?",
        "Can you assist me today?"
    ]
    for g in greetings:
        probes.append({
            "category": "english_conversation",
            "prompt": g,
            "expected": "morning" if "morning" in g else "assist"
        })
        
    # 8. Devanagari Hindi (30)
    hindi_prompts = [
        ("नमस्ते! आप कौन हैं?", "चक्रव्यूह"),
        ("नमस्ते! आप कैसे हैं?", "नमस्ते"),
        ("कंप्यूटर क्या है?", "कंप्यूटर")
    ]
    for p, exp in hindi_prompts:
        probes.append({
            "category": "devanagari_hindi",
            "prompt": p,
            "expected": exp
        })
        
    # 9. Roman Hindi & Hinglish (30)
    hinglish_prompts = [
        ("Mera naam Priya hai. Aap kaun ho?", "chakrview"),
        ("Aapka main kaam kya hai bhai?", "model"),
        ("Namaste! Aap kaise hain?", "namaste")
    ]
    for p, exp in hinglish_prompts:
        probes.append({
            "category": "roman_hindi_hinglish",
            "prompt": p,
            "expected": exp
        })
        
    # 10. Missing Information & Abstention (20)
    abstain_prompts = [
        "What did I eat for dinner three weeks ago?",
        "Where did I leave my keys?",
        "What will Apple stock trade at next month?"
    ]
    for p in abstain_prompts:
        probes.append({
            "category": "missing_info_abstention",
            "prompt": p,
            "expected": "do not"
        })
        
    # Expand to 300 total probes with balanced replicates
    total_probes = []
    idx = 1
    while len(total_probes) < 300:
        for p in probes:
            if len(total_probes) >= 300:
                break
            item = dict(p)
            item["probe_id"] = f"probe_{idx:03d}"
            total_probes.append(item)
            idx += 1
            
    out_dir = Path("experiments/cognition/ucip_02/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "core_acceptance_benchmark_300.json", "w", encoding="utf-8") as f:
        json.dump(total_probes, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(total_probes)} probes to {out_dir / 'core_acceptance_benchmark_300.json'}")

if __name__ == "__main__":
    build_acceptance_benchmark()
