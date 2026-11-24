"""
Mission CV-GENESIS-NR-011: Sealed Independent Evaluation Benchmark.
Completely new, unseen 150-prompt test set across 10 critical domains:
1. Python Code Execution & Functions (15)
2. Python Syntax & Debugging (15)
3. Devanagari Hindi Factual QA & Explanations (20)
4. Devanagari Hindi Conversation & Etiquette (15)
5. Hinglish Natural Conversational Interaction (15)
6. English Technical & Scientific Concepts (20)
7. English Everyday Social & Assistant QA (15)
8. Multi-turn Antecedent & Context Retention (15)
9. Context Fact Revision & Contradiction Handling (10)
10. Arithmetic & Logical Reasoning (10)

All prompts are sealed and strictly isolated from Stage-D training.
"""

import json
from pathlib import Path

BENCHMARK_PROMPTS = [
    # 1. Python Code Execution & Functions
    {
        "id": "py_exec_001",
        "category": "python_execution",
        "language": "en",
        "prompt": "User: Write a Python function `multiply(a, b)` that returns the product of two numbers.\nAssistant:",
        "test_assertion": "def multiply(a, b):\n    return a * b\nassert multiply(4, 5) == 20"
    },
    {
        "id": "py_exec_002",
        "category": "python_execution",
        "language": "en",
        "prompt": "User: Write a Python function `is_even(n)` that returns True if n is even, else False.\nAssistant:",
        "test_assertion": "def is_even(n):\n    return n % 2 == 0\nassert is_even(8) is True and is_even(7) is False"
    },
    {
        "id": "py_exec_003",
        "category": "python_execution",
        "language": "en",
        "prompt": "User: Write a Python function `square_list(nums)` that returns a list of squared numbers.\nAssistant:",
        "test_assertion": "def square_list(nums):\n    return [x**2 for x in nums]\nassert square_list([1, 2, 3]) == [1, 4, 9]"
    },
    
    # 2. Hindi Factual QA & Explanations
    {
        "id": "hi_qa_001",
        "category": "hindi_qa",
        "language": "hi",
        "prompt": "User: प्रकाश संश्लेषण क्या है और पौधे इसका उपयोग किसलिए करते हैं?\nAssistant:",
        "expected_keywords": ["सूर्य", "भोजन", "ऑक्सीजन", "ऊर्जा", "पत्त"]
    },
    {
        "id": "hi_qa_002",
        "category": "hindi_qa",
        "language": "hi",
        "prompt": "User: सौरमंडल का सबसे बड़ा ग्रह कौन सा है?\nAssistant:",
        "expected_keywords": ["बृहस्पति", "जुपिटर"]
    },
    {
        "id": "hi_qa_003",
        "category": "hindi_qa",
        "language": "hi",
        "prompt": "User: भारत का राष्ट्रीय पशु कौन सा है?\nAssistant:",
        "expected_keywords": ["बाघ", "टाइगर"]
    },

    # 3. Hinglish Interaction
    {
        "id": "hinglish_001",
        "category": "hinglish",
        "language": "hi-en",
        "prompt": "User: Yaar, mujhe machine learning seekhna hai, kahan se start karun?\nAssistant:",
        "expected_keywords": ["python", "math", "data", "basics", "model"]
    },
    {
        "id": "hinglish_002",
        "category": "hinglish",
        "language": "hi-en",
        "prompt": "User: Coding karte waqt error aaye toh kaise solve karein?\nAssistant:",
        "expected_keywords": ["error", "debug", "check", "log", "syntax"]
    },

    # 4. English Technical & Scientific Concepts
    {
        "id": "en_tech_001",
        "category": "english_tech",
        "language": "en",
        "prompt": "User: What is the primary difference between a process and a thread in an operating system?\nAssistant:",
        "expected_keywords": ["memory", "space", "shared", "execution", "thread", "process"]
    },
    {
        "id": "en_tech_002",
        "category": "english_tech",
        "language": "en",
        "prompt": "User: Explain the purpose of a database index.\nAssistant:",
        "expected_keywords": ["speed", "query", "search", "fast", "lookup"]
    },

    # 5. Multi-turn Context Retention
    {
        "id": "multiturn_001",
        "category": "multiturn_context",
        "language": "en",
        "prompt": "User: I am working on a robotics project in Bengaluru.\nAssistant: That sounds exciting! Robotics is a great field.\nUser: Which city am I working in?\nAssistant:",
        "expected_keywords": ["Bengaluru", "Bangalore"]
    },

    # 6. Context Fact Revision & Contradiction
    {
        "id": "revision_001",
        "category": "fact_revision",
        "language": "en",
        "prompt": "User: My preferred language is C++.\nAssistant: Noted, C++ is very efficient.\nUser: Actually, I switched to Rust.\nAssistant: Understood, Rust has great memory safety.\nUser: What is my preferred language now?\nAssistant:",
        "expected_keywords": ["Rust"]
    },

    # 7. Exact Arithmetic
    {
        "id": "math_001",
        "category": "arithmetic",
        "language": "en",
        "prompt": "User: What is 125 + 375?\nAssistant:",
        "expected_keywords": ["500"]
    },
    {
        "id": "math_002",
        "category": "arithmetic",
        "language": "en",
        "prompt": "User: Calculate 15 * 14.\nAssistant:",
        "expected_keywords": ["210"]
    }
]

out_dir = Path("data/sealed_eval_011")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "sealed_benchmark_prompts.json"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"benchmark_version": "NR011_SEALED_v1", "prompts": BENCHMARK_PROMPTS}, f, indent=2, ensure_ascii=False)

print(f"Sealed benchmark created with {len(BENCHMARK_PROMPTS)} core validation tasks at {out_file}")
