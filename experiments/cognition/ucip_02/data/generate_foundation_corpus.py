import random
import json
from pathlib import Path

random.seed(42)

def build_foundation_conversational_corpus():
    records = []
    
    # 1. Exact sentence copying & repetition
    sentences = [
        "The sky is blue.",
        "A dog is man's best friend.",
        "Water boils at 100 degrees Celsius.",
        "ChakrView is an indigenous neural language model.",
        "Python is a popular programming language.",
        "The Earth revolves around the Sun.",
        "Consistency and honesty are essential in science.",
        "The capital of France is Paris."
    ]
    for s in sentences:
        records.append({
            "prompt": f"User: Say exactly: {s}\nAssistant:",
            "target": f" {s}",
            "category": "exact_copying"
        })
        records.append({
            "prompt": f"User: Repeat after me: {s}\nAssistant:",
            "target": f" {s}",
            "category": "exact_copying"
        })
        
    # 2. Simple Sentence Generation
    topics = [
        ("a dog", "A dog is a loyal and playful companion."),
        ("the sun", "The sun provides warmth and light to our planet."),
        ("a tree", "A tree provides oxygen and shade to the environment."),
        ("a computer", "A computer processes instructions to perform useful tasks."),
        ("a book", "A book contains knowledge and stories to read.")
    ]
    for topic, sentence in topics:
        records.append({
            "prompt": f"User: Write one simple sentence about {topic}.\nAssistant:",
            "target": f" {sentence}",
            "category": "simple_generation"
        })
        records.append({
            "prompt": f"User: Tell me a simple sentence about {topic}.\nAssistant:",
            "target": f" {sentence}",
            "category": "simple_generation"
        })

    # 3. Basic Arithmetic
    for a in range(1, 15):
        for b in range(1, 15):
            records.append({
                "prompt": f"User: What is {a} + {b}?\nAssistant:",
                "target": f" {a} + {b} is {a + b}.",
                "category": "basic_arithmetic"
            })
            if a >= b:
                records.append({
                    "prompt": f"User: What is {a} - {b}?\nAssistant:",
                    "target": f" {a} - {b} is {a - b}.",
                    "category": "basic_arithmetic"
                })

    # 4. Hybrid Memory Conditioning (Crucial for Name & Fact Recall!)
    names = ["Rahul", "Priya", "Rohan", "Neha", "Kabir", "Ananya", "Vikram", "Sneha", "Amit", "Pooja"]
    cities = ["Pune", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Jaipur", "Patna", "Kolkata"]
    projects = ["ChakrView", "Yatrajivi", "Drishti", "Setu", "Kavach", "Bodhi"]
    
    for n in names:
        # Prompt with [Memory: Name: {n}]
        records.append({
            "prompt": f"[Memory: Name: {n}]\nUser: What is my name?\nAssistant:",
            "target": f" Your name is {n}.",
            "category": "memory_conditioned_name"
        })
        records.append({
            "prompt": f"[Memory: Name: {n}]\nUser: Do you know my name?\nAssistant:",
            "target": f" Yes, your name is {n}.",
            "category": "memory_conditioned_name"
        })
        records.append({
            "prompt": f"[Memory: Name: {n}]\nUser: Mera naam kya hai?\nAssistant:",
            "target": f" Aapka naam {n} hai.",
            "category": "memory_conditioned_name"
        })
        # Natural multi-turn dialogue with memory prefix
        records.append({
            "prompt": f"User: My name is {n}.\nAssistant: Hello {n}! How may I help you today?\n[Memory: Name: {n}]\nUser: What is my name?\nAssistant:",
            "target": f" Your name is {n}.",
            "category": "memory_multiturn"
        })

    for c in cities:
        records.append({
            "prompt": f"[Memory: City: {c}]\nUser: Where do I live?\nAssistant:",
            "target": f" You live in {c}.",
            "category": "memory_conditioned_city"
        })
        records.append({
            "prompt": f"[Memory: City: {c}]\nUser: Which city am I located in?\nAssistant:",
            "target": f" You are located in {c}.",
            "category": "memory_conditioned_city"
        })

    for p in projects:
        records.append({
            "prompt": f"[Memory: Project: {p}]\nUser: What is my project called?\nAssistant:",
            "target": f" Your project is called {p}.",
            "category": "memory_conditioned_project"
        })
        records.append({
            "prompt": f"[Memory: Project: {p}]\nUser: What am I working on?\nAssistant:",
            "target": f" You are working on {p}.",
            "category": "memory_conditioned_project"
        })

    # 5. Natural Persona, Identity, Greetings
    records.append({
        "prompt": "User: Who are you?\nAssistant:",
        "target": " I am ChakrView, an indigenous neural language model developed from first principles.",
        "category": "persona"
    })
    records.append({
        "prompt": "User: What is your name?\nAssistant:",
        "target": " I am ChakrView, an indigenous neural conversational assistant.",
        "category": "persona"
    })
    records.append({
        "prompt": "User: Hello!\nAssistant:",
        "target": " Hello! How can I assist you today?",
        "category": "greeting"
    })
    records.append({
        "prompt": "User: Good morning assistant!\nAssistant:",
        "target": " Good morning! How can I help you today?",
        "category": "greeting"
    })

    # 6. Devanagari Hindi & Hinglish
    records.append({
        "prompt": "User: नमस्ते! आप कौन हैं?\nAssistant:",
        "target": " नमस्ते! मैं चक्रव्यूह हूँ, एक स्वदेशी न्यूरल भाषा मॉडल।",
        "category": "hindi_devanagari"
    })
    records.append({
        "prompt": "User: नमस्ते! आप कैसे हैं?\nAssistant:",
        "target": " नमस्ते! मैं ठीक हूँ। मैं आपकी क्या मदद कर सकता हूँ?",
        "category": "hindi_devanagari"
    })
    records.append({
        "prompt": "User: Aapka naam kya hai bhai?\nAssistant:",
        "target": " Mera naam ChakrView hai, main ek indigenous neural model hoon.",
        "category": "hinglish"
    })
    records.append({
        "prompt": "User: Namaste! Aap kaise hain?\nAssistant:",
        "target": " Namaste! Main badhiya hoon, aap bataiye main aapki kya madad kar sakta hoon?",
        "category": "hinglish"
    })

    # 7. Uncertainty & Abstention
    unknown_queries = [
        "What did I eat for dinner three weeks ago?",
        "Where did I leave my keys?",
        "What is my password?",
        "What will the stock market do tomorrow?"
    ]
    for q in unknown_queries:
        records.append({
            "prompt": f"User: {q}\nAssistant:",
            "target": " I do not have access to that information.",
            "category": "abstention"
        })

    # Duplicate / balance to ~1,500 records
    extended_records = []
    for _ in range(3):
        for r in records:
            extended_records.append(dict(r))
            
    random.shuffle(extended_records)
    
    out_dir = Path("experiments/cognition/ucip_02/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    split = int(0.9 * len(extended_records))
    train_data = extended_records[:split]
    val_data = extended_records[split:]
    
    with open(out_dir / "ucip02_train.jsonl", "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(out_dir / "ucip02_val.jsonl", "w", encoding="utf-8") as f:
        for r in val_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(train_data)} train records and {len(val_data)} val records for UCIP-02.")

if __name__ == "__main__":
    build_foundation_conversational_corpus()
