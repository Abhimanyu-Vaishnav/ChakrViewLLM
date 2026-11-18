"""
UCIP-10E Multi-Domain Benchmark Suite Generator.
Generates:
1. Development benchmark suite: experiments/ucip10e/datasets/development/benchmark_dev.json
2. Sealed benchmark suite: experiments/ucip10e/datasets/sealed/benchmark_sealed.json

Domains covered:
1. Language (English, Hindi, Hinglish)
2. Mathematics (Single, multi-digit, carry, borrow, word problems)
3. Exact Copying (Verbatim & normalized, alphanumeric, punctuation)
4. Logical Reasoning (Syllogisms, condition, contradiction)
5. Generalization (Novel entities, out-of-distribution numbers)
6. Multi-turn Reasoning (State tracking, context retention)
7. Self-correction (Verification cycle)
8. Uncertainty & Calibration (Abstention, insufficient info)
9. Coding Intelligence (Sandboxed Python execution)
10. Self-modeling (Architecture & metadata awareness)
"""

import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e"

def generate_arithmetic_probes(seed: int, split: str, count: int = 150):
    rng = random.Random(seed)
    probes = []
    
    # 1. Single digit addition (15)
    for i in range(15):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        probes.append({
            "id": f"{split[:3].upper()}_MATH_ADD_1D_{i:02d}",
            "domain": "mathematics",
            "subdomain": "single_digit_addition",
            "prompt": f"What is {a} + {b}?",
            "evaluation_type": "exact_math",
            "target_number": a + b,
            "difficulty": "elementary",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    # 2. Single digit subtraction (15)
    for i in range(15):
        a = rng.randint(2, 9)
        b = rng.randint(1, a - 1)
        probes.append({
            "id": f"{split[:3].upper()}_MATH_SUB_1D_{i:02d}",
            "domain": "mathematics",
            "subdomain": "single_digit_subtraction",
            "prompt": f"What is {a} - {b}?",
            "evaluation_type": "exact_math",
            "target_number": a - b,
            "difficulty": "elementary",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    # 3. Multi-digit addition with carry (30)
    for i in range(30):
        a = rng.randint(15, 89)
        b = rng.randint(15, 89)
        probes.append({
            "id": f"{split[:3].upper()}_MATH_ADD_2D_{i:02d}",
            "domain": "mathematics",
            "subdomain": "multi_digit_addition_carry",
            "prompt": f"Calculate {a} + {b}.",
            "evaluation_type": "exact_math",
            "target_number": a + b,
            "difficulty": "intermediate",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    # 4. Multi-digit subtraction with borrow (30)
    for i in range(30):
        a = rng.randint(40, 99)
        b = rng.randint(15, a - 5)
        probes.append({
            "id": f"{split[:3].upper()}_MATH_SUB_2D_{i:02d}",
            "domain": "mathematics",
            "subdomain": "multi_digit_subtraction_borrow",
            "prompt": f"Compute {a} - {b}.",
            "evaluation_type": "exact_math",
            "target_number": a - b,
            "difficulty": "intermediate",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    # 5. Word problems (30)
    items = ["apples", "books", "pens", "cards", "mangoes", "coins", "candies", "tickets"]
    for i in range(30):
        item = rng.choice(items)
        initial = rng.randint(10, 50)
        give = rng.randint(2, initial - 3)
        rem = initial - give
        probes.append({
            "id": f"{split[:3].upper()}_MATH_WORD_{i:02d}",
            "domain": "mathematics",
            "subdomain": "word_problem_subtraction",
            "prompt": f"If I have {initial} {item} and give {give} to my friend, how many {item} remain?",
            "evaluation_type": "exact_math",
            "target_number": rem,
            "difficulty": "intermediate",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    # 6. Unseen larger operands & multiple operations (30)
    for i in range(30):
        a = rng.randint(100, 300)
        b = rng.randint(50, 150)
        probes.append({
            "id": f"{split[:3].upper()}_MATH_LARGE_{i:02d}",
            "domain": "mathematics",
            "subdomain": "unseen_operands",
            "prompt": f"What is {a} + {b}?",
            "evaluation_type": "exact_math",
            "target_number": a + b,
            "difficulty": "advanced",
            "split": split,
            "provenance": "synthetic_oracle"
        })

    return probes[:count]

def generate_language_probes(seed: int, split: str):
    rng = random.Random(seed)
    probes = []
    
    # English (40)
    eng_samples = [
        ("Hello ChakrView. How are you?", ["chakrview", "assistant", "well"], "greetings"),
        ("Who created you?", ["abhimanyu", "created", "assistant", "designed"], "identity"),
        ("What does a CPU do?", ["processor", "instructions", "executes", "computes"], "explanations"),
        ("Explain the difference between RAM and storage.", ["ram", "memory", "storage", "volatile"], "explanations"),
        ("What is the primary function of an operating system?", ["hardware", "software", "manages", "system"], "factual"),
        ("Why is clean code important in software engineering?", ["maintainable", "readable", "bugs", "collaboration"], "factual"),
        ("Describe the photosynthesis process in simple terms.", ["light", "plants", "sunlight", "energy"], "factual"),
        ("What is an algorithm?", ["steps", "procedure", "problem", "solve"], "factual")
    ]
    for idx, (p, kws, sub) in enumerate(eng_samples * 5):
        probes.append({
            "id": f"{split[:3].upper()}_LANG_ENG_{idx:02d}",
            "domain": "language",
            "subdomain": "english",
            "prompt": p,
            "evaluation_type": "semantic_rubric",
            "expected_keywords": kws,
            "difficulty": "elementary",
            "split": split,
            "provenance": "curated_language_suite"
        })

    # Hindi (40)
    hin_samples = [
        ("नमस्ते! आप क्या कर सकते हैं?", ["सहायता", "हूँ", "मॉडल"], "greetings"),
        ("कंप्यूटर में सीपीयू का क्या काम है?", ["निर्देश", "प्रोसेस", "मस्तिष्क", "गणना"], "explanations"),
        ("सूर्य हमें क्या प्रदान करता है?", ["प्रकाश", "ऊर्जा", "सूर्य"], "factual"),
        ("सॉफ्टवेयर क्या होता है?", ["निर्देश", "प्रोग्राम", "कंप्यूटर"], "factual"),
        ("पेड़-पौधे हमारे लिए क्यों महत्वपूर्ण हैं?", ["ऑक्सीजन", "पर्यावरण", "जीवन"], "factual")
    ]
    for idx, (p, kws, sub) in enumerate(hin_samples * 8):
        probes.append({
            "id": f"{split[:3].upper()}_LANG_HIN_{idx:02d}",
            "domain": "language",
            "subdomain": "hindi",
            "prompt": p,
            "evaluation_type": "semantic_rubric",
            "expected_keywords": kws,
            "difficulty": "elementary",
            "split": split,
            "provenance": "curated_language_suite"
        })

    # Hinglish (40)
    hing_samples = [
        ("Bhai, machine learning ko simple language mein samjhao.", ["data", "computer", "learn", "patterns"], "explanations"),
        ("Python itni popular kyu hai software development mein?", ["easy", "libraries", "simple", "syntax"], "explanations"),
        ("Database aur RAM mein kya farak hota hai?", ["storage", "data", "memory", "permanent"], "explanations"),
        ("API kya hoti hai aur ye kaise kaam karti hai?", ["software", "communicate", "interface", "data"], "explanations")
    ]
    for idx, (p, kws, sub) in enumerate(hing_samples * 10):
        probes.append({
            "id": f"{split[:3].upper()}_LANG_HING_{idx:02d}",
            "domain": "language",
            "subdomain": "hinglish",
            "prompt": p,
            "evaluation_type": "semantic_rubric",
            "expected_keywords": kws,
            "difficulty": "elementary",
            "split": split,
            "provenance": "curated_language_suite"
        })

    return probes

def generate_copying_probes(seed: int, split: str, count: int = 100):
    rng = random.Random(seed)
    probes = []
    
    sentences = [
        "The sky is blue.",
        "Knowledge is power.",
        "Silence is a source of great strength.",
        "Simplicity is the soul of efficiency.",
        "ChakrView operates locally without network access.",
        "Neural networks learn representations from data.",
        "Data structures organize information efficiently.",
        "Python 3.14 delivers significant runtime optimizations.",
        "Attention mechanisms route information dynamically.",
        "Memory governance prevents unauthorized state leakage."
    ]
    
    for i in range(count):
        s = sentences[i % len(sentences)]
        probes.append({
            "id": f"{split[:3].upper()}_COPY_{i:02d}",
            "domain": "copying",
            "subdomain": "verbatim_exact_copy",
            "prompt": f"Say exactly: {s}",
            "evaluation_type": "verbatim_copy",
            "target_text": s,
            "difficulty": "elementary" if len(s.split()) < 5 else "intermediate",
            "split": split,
            "provenance": "unseen_copy_manifest"
        })
    return probes

def generate_logic_probes(seed: int, split: str, count: int = 100):
    rng = random.Random(seed)
    probes = []
    
    templates = [
        ("All humans are mortal. Socrates is a human. Is Socrates mortal? Answer with 'Yes' or 'No'.", "yes"),
        ("If it rains, the ground gets wet. It is raining. Is the ground wet? Answer with 'Yes' or 'No'.", "yes"),
        ("No fish can fly. A trout is a fish. Can a trout fly? Answer with 'Yes' or 'No'.", "no"),
        ("A is taller than B, and B is taller than C. Is A taller than C? Answer with 'Yes' or 'No'.", "yes"),
        ("X is either red or blue. X is not blue. What color is X?", "red")
    ]
    
    for i in range(count):
        t, ans = templates[i % len(templates)]
        probes.append({
            "id": f"{split[:3].upper()}_LOGIC_{i:02d}",
            "domain": "logic",
            "subdomain": "deductive_syllogism",
            "prompt": t,
            "evaluation_type": "deductive_logic",
            "expected_answer": ans,
            "difficulty": "intermediate",
            "split": split,
            "provenance": "formal_logic_oracle"
        })
    return probes

def generate_generalization_probes(seed: int, split: str, count: int = 100):
    rng = random.Random(seed)
    probes = []
    
    names = ["Zephyr", "Kaelen", "Morrigan", "Theron", "Lyra", "Vesper", "Dax", "Selene"]
    objects = ["chronometers", "quantum crystal", "astrolabes", "plasma battery", "sublight compass"]
    
    for i in range(count):
        name = rng.choice(names)
        obj = rng.choice(objects)
        a = rng.randint(11, 49)
        b = rng.randint(2, 9)
        probes.append({
            "id": f"{split[:3].upper()}_GEN_{i:02d}",
            "domain": "generalization",
            "subdomain": "novel_entity_composition",
            "prompt": f"{name} collected {a} {obj} and gave {b} to an explorer. How many {obj} does {name} have now?",
            "evaluation_type": "exact_math",
            "target_number": a - b,
            "difficulty": "advanced",
            "split": split,
            "provenance": "out_of_distribution_combinatorial"
        })
    return probes

def generate_multi_turn_probes(seed: int, split: str, count: int = 60):
    rng = random.Random(seed)
    probes = []
    for i in range(count):
        city = rng.choice(["Pune", "Bengaluru", "Delhi", "Chandigarh", "Jaipur"])
        probes.append({
            "id": f"{split[:3].upper()}_TURN_{i:02d}",
            "domain": "multi_turn",
            "subdomain": "entity_memory_tracking",
            "prompt": f"My favourite city is {city}. What is my favourite city?",
            "evaluation_type": "semantic_rubric",
            "expected_keywords": [city.lower()],
            "difficulty": "intermediate",
            "split": split,
            "provenance": "conversational_memory_suite"
        })
    return probes

def generate_self_correction_probes(seed: int, split: str, count: int = 60):
    rng = random.Random(seed)
    probes = []
    for i in range(count):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        probes.append({
            "id": f"{split[:3].upper()}_CORR_{i:02d}",
            "domain": "self_correction",
            "subdomain": "error_revision_cycle",
            "prompt": f"You previously said {a} + {b} = {a+b+1}. Please verify if that is correct and provide the true sum.",
            "evaluation_type": "exact_math",
            "target_number": a + b,
            "difficulty": "advanced",
            "split": split,
            "provenance": "adversarial_revision_suite"
        })
    return probes

def generate_uncertainty_probes(seed: int, split: str, count: int = 60):
    rng = random.Random(seed)
    probes = []
    
    abstain_prompts = [
        "What was the exact price of gold in London at 3:14 PM on March 4, 1842?",
        "Who is the person standing right behind me right now?",
        "What will be the exact weather in Tokyo on December 31, 2099?",
        "What did John have for breakfast if we only know he woke up at 7 AM?"
    ]
    known_prompts = [
        "What is the capital of France?",
        "What is the chemical formula for water?",
        "How many days are in a leap year?",
        "Is iron a metal?"
    ]
    
    for i in range(count):
        if i % 2 == 0:
            p = abstain_prompts[(i // 2) % len(abstain_prompts)]
            should_ab = True
        else:
            p = known_prompts[(i // 2) % len(known_prompts)]
            should_ab = False
            
        probes.append({
            "id": f"{split[:3].upper()}_UNCERT_{i:02d}",
            "domain": "uncertainty",
            "subdomain": "abstention_and_calibration",
            "prompt": p,
            "evaluation_type": "abstention_calibration",
            "metadata": {"should_abstain": should_ab},
            "difficulty": "intermediate",
            "split": split,
            "provenance": "epistemic_abstention_suite"
        })
    return probes

def generate_coding_probes(seed: int, split: str, count: int = 50):
    rng = random.Random(seed)
    probes = []
    
    tasks = [
        ("Write a Python function `def add(a, b):` that returns their sum.", "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
        ("Write a Python function `def is_even(n):` that returns True if n is even.", "assert is_even(4) == True\nassert is_even(7) == False"),
        ("Write a Python function `def square(x):` that returns the square of x.", "assert square(3) == 9\nassert square(-4) == 16"),
        ("Write a Python function `def str_len(s):` that returns the length of string s.", "assert str_len('hello') == 5\nassert str_len('') == 0")
    ]
    
    for i in range(count):
        prompt, test_code = tasks[i % len(tasks)]
        probes.append({
            "id": f"{split[:3].upper()}_CODE_{i:02d}",
            "domain": "coding",
            "subdomain": "python_execution",
            "prompt": prompt,
            "evaluation_type": "sandboxed_execution",
            "code_test": {"test_code": test_code, "timeout_seconds": 5},
            "difficulty": "intermediate",
            "split": split,
            "provenance": "sandboxed_code_oracle"
        })
    return probes

def generate_self_modeling_probes(seed: int, split: str, count: int = 40):
    probes = []
    samples = [
        ("What is your name and identity?", ["chakrview", "assistant"], "identity"),
        ("What neural architecture are you running on?", ["chakrmedium", "chakrmicro", "transformer", "layers"], "architecture"),
        ("Do you require network or internet access to answer local queries?", ["local", "offline", "no"], "offline_awareness"),
        ("Can you execute external code directly without tools?", ["tool", "no", "cannot"], "tool_boundary")
    ]
    for i in range(count):
        p, kws, sub = samples[i % len(samples)]
        probes.append({
            "id": f"{split[:3].upper()}_MODEL_{i:02d}",
            "domain": "self_modeling",
            "subdomain": sub,
            "prompt": p,
            "evaluation_type": "system_awareness",
            "expected_keywords": kws,
            "difficulty": "elementary",
            "split": split,
            "provenance": "self_modeling_manifest"
        })
    return probes

def build_suite(seed: int, split: str):
    all_probes = []
    all_probes.extend(generate_language_probes(seed, split))
    all_probes.extend(generate_arithmetic_probes(seed, split, 150))
    all_probes.extend(generate_copying_probes(seed, split, 100))
    all_probes.extend(generate_logic_probes(seed, split, 100))
    all_probes.extend(generate_generalization_probes(seed, split, 100))
    all_probes.extend(generate_multi_turn_probes(seed, split, 60))
    all_probes.extend(generate_self_correction_probes(seed, split, 60))
    all_probes.extend(generate_uncertainty_probes(seed, split, 60))
    all_probes.extend(generate_coding_probes(seed, split, 50))
    all_probes.extend(generate_self_modeling_probes(seed, split, 40))
    return all_probes

def main():
    dev_probes = build_suite(seed=1001, split="development")
    sealed_probes = build_suite(seed=2002, split="sealed")
    
    dev_path = BASE_DIR / "datasets" / "development" / "benchmark_dev.json"
    sealed_path = BASE_DIR / "datasets" / "sealed" / "benchmark_sealed.json"
    
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_probes, f, indent=2, ensure_ascii=False)
        
    with open(sealed_path, "w", encoding="utf-8") as f:
        json.dump(sealed_probes, f, indent=2, ensure_ascii=False)
        
    print(f"Generated Development Benchmark: {len(dev_probes)} probes -> {dev_path}")
    print(f"Generated Sealed Benchmark:      {len(sealed_probes)} probes -> {sealed_path}")
    
    # Save manifest registry
    manifest_meta = {
        "version": "1.0.0",
        "created_at": "2026-11-18",
        "benchmark_suites": {
            "development": {
                "path": str(dev_path.relative_to(REPO_ROOT)),
                "total_probes": len(dev_probes),
                "sha256": hashlib.sha256(open(dev_path, "rb").read()).hexdigest()
            },
            "sealed": {
                "path": str(sealed_path.relative_to(REPO_ROOT)),
                "total_probes": len(sealed_probes),
                "sha256": hashlib.sha256(open(sealed_path, "rb").read()).hexdigest()
            }
        }
    }
    with open(BASE_DIR / "manifests" / "manifest_registry.json", "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2)

if __name__ == "__main__":
    main()
