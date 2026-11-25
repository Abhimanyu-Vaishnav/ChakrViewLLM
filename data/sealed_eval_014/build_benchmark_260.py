"""
ChakrView Fresh Sealed Generalization Benchmark Suite (260 Tasks).
Mission CV-GENESIS-NR-014 Phase 5.

Categories (260 Tasks):
1. Python Function Generation (40 tasks) - evaluated via sandboxed subprocess execution with hidden test cases
2. Python Debugging & Explanation (20 tasks)
3. Algorithmic Reasoning (20 tasks)
4. Arithmetic & Exact Calculations (25 tasks)
5. English Conversational Comprehension (30 tasks)
6. Hindi Conversational Comprehension (30 tasks)
7. Hinglish Dialogue (30 tasks)
8. Instruction Following & Output Constraints (25 tasks)
9. Multi-Turn Context & Antecedent Retention (20 tasks)
10. Safety Refusal & Principled Abstention (20 tasks)
"""

from __future__ import annotations
import json
from pathlib import Path

# Load existing 210 benchmark items
old_bench_p = Path("data/sealed_eval_013/sealed_benchmark_210.json")
with open(old_bench_p, "r", encoding="utf-8") as f:
    benchmark_items = json.load(f)

# Add 50 fresh unseen tasks across coding, arithmetic, edge cases, and reasoning
fresh_tasks = [
    # 10 Additional Python Codegen Tasks with Subprocess Tests
    {"id": "py_gen_31", "category": "python_codegen", "prompt": "User: Write a python function `is_even_integer(n)` returning True if n is even else False.\nAssistant:", "test_code": "assert is_even_integer(8) is True and is_even_integer(3) is False"},
    {"id": "py_gen_32", "category": "python_codegen", "prompt": "User: Write a python function `multiply_pair(a, b)` returning their product.\nAssistant:", "test_code": "assert multiply_pair(11, 4) == 44 and multiply_pair(-2, 8) == -16"},
    {"id": "py_gen_33", "category": "python_codegen", "prompt": "User: Write a python function `sum_pair(a, b)` returning their sum.\nAssistant:", "test_code": "assert sum_pair(25, 35) == 60 and sum_pair(-10, 10) == 0"},
    {"id": "py_gen_34", "category": "python_codegen", "prompt": "User: Write a python function `string_length_int(s)` returning the length of string s.\nAssistant:", "test_code": "assert string_length_int('deepmind') == 8 and string_length_int('') == 0"},
    {"id": "py_gen_35", "category": "python_codegen", "prompt": "User: Write a python function `negate_number(x)` returning negative of x.\nAssistant:", "test_code": "assert negate_number(15) == -15 and negate_number(-4) == 4"},
    {"id": "py_gen_36", "category": "python_codegen", "prompt": "User: Write a python function `cube_integer(n)` returning n to the third power.\nAssistant:", "test_code": "assert cube_integer(5) == 125 and cube_integer(-3) == -27"},
    {"id": "py_gen_37", "category": "python_codegen", "prompt": "User: Write a python function `square_integer(n)` returning n squared.\nAssistant:", "test_code": "assert square_integer(12) == 144 and square_integer(-5) == 25"},
    {"id": "py_gen_38", "category": "python_codegen", "prompt": "User: Write a python function `first_list_val(lst)` returning the first element of a non-empty list.\nAssistant:", "test_code": "assert first_list_val([77, 88, 99]) == 77"},
    {"id": "py_gen_39", "category": "python_codegen", "prompt": "User: Write a python function `last_list_val(lst)` returning the last element of a non-empty list.\nAssistant:", "test_code": "assert last_list_val([77, 88, 99]) == 99"},
    {"id": "py_gen_40", "category": "python_codegen", "prompt": "User: Write a python function `is_list_empty_check(lst)` returning True if list is empty else False.\nAssistant:", "test_code": "assert is_list_empty_check([]) is True and is_list_empty_check([1]) is False"},

    # 5 Additional Explanation & Debugging
    {"id": "dbg_16", "category": "python_explanation", "prompt": "User: Why is using a mutable default argument like `def func(lst=[])` dangerous in Python?\nAssistant:", "keywords": ["mutable", "default", "shared", "persists", "calls"]},
    {"id": "dbg_17", "category": "python_explanation", "prompt": "User: What does AttributeError mean in Python?\nAssistant:", "keywords": ["attribute", "object", "exist", "method", "type"]},
    {"id": "dbg_18", "category": "python_explanation", "prompt": "User: How does the `with` statement work in Python file handling?\nAssistant:", "keywords": ["context", "close", "cleanup", "file", "closes"]},
    {"id": "dbg_19", "category": "python_explanation", "prompt": "User: What is the difference between shallow copy and deep copy?\nAssistant:", "keywords": ["nested", "reference", "independent", "copy", "objects"]},
    {"id": "dbg_20", "category": "python_explanation", "prompt": "User: What is a generator function and how does `yield` work?\nAssistant:", "keywords": ["yield", "iterator", "generator", "memory", "state"]},

    # 5 Additional Algorithmic Tasks
    {"id": "algo_16", "category": "algorithmic_reasoning", "prompt": "User: What is the time complexity of appending an element to a Python list on average?\nAssistant:", "keywords": ["o(1)", "constant", "amortized", "append"]},
    {"id": "algo_17", "category": "algorithmic_reasoning", "prompt": "User: What is a palindrome?\nAssistant:", "keywords": ["same", "forward", "backward", "reverse", "reads"]},
    {"id": "algo_18", "category": "algorithmic_reasoning", "prompt": "User: What is recursion depth limit in programming?\nAssistant:", "keywords": ["stack", "overflow", "limit", "recursion", "depth"]},
    {"id": "algo_19", "category": "algorithmic_reasoning", "prompt": "User: Why is QuickSort often preferred over MergeSort in-memory?\nAssistant:", "keywords": ["in-place", "cache", "memory", "space", "auxiliary"]},
    {"id": "algo_20", "category": "algorithmic_reasoning", "prompt": "User: What is the time complexity of accessing an element in an array by index?\nAssistant:", "keywords": ["o(1)", "constant", "direct", "index"]},

    # 5 Additional Arithmetic
    {"id": "math_21", "category": "arithmetic", "prompt": "User: Calculate 150 + 250\nAssistant:", "expected": "400"},
    {"id": "math_22", "category": "arithmetic", "prompt": "User: Calculate 90 - 45\nAssistant:", "expected": "45"},
    {"id": "math_23", "category": "arithmetic", "prompt": "User: Calculate 18 * 4\nAssistant:", "expected": "72"},
    {"id": "math_24", "category": "arithmetic", "prompt": "User: Calculate 300 / 15\nAssistant:", "expected": "20"},
    {"id": "math_25", "category": "arithmetic", "prompt": "User: Calculate 85 + 15\nAssistant:", "expected": "100"},

    # 5 English Conv
    {"id": "en_26", "category": "english_conv", "prompt": "User: What is an algorithm?\nAssistant:", "keywords": ["step", "procedure", "instructions", "solve", "problem"]},
    {"id": "en_27", "category": "english_conv", "prompt": "User: What is currency inflation?\nAssistant:", "keywords": ["prices", "money", "purchasing", "value", "increase"]},
    {"id": "en_28", "category": "english_conv", "prompt": "User: How does the internet work?\nAssistant:", "keywords": ["network", "computers", "data", "servers", "packets"]},
    {"id": "en_29", "category": "english_conv", "prompt": "User: What is machine learning?\nAssistant:", "keywords": ["data", "algorithms", "learning", "models", "patterns"]},
    {"id": "en_30", "category": "english_conv", "prompt": "User: What is a black hole?\nAssistant:", "keywords": ["gravity", "light", "space", "star", "dense"]},

    # 5 Hindi Conv
    {"id": "hi_26", "category": "hindi_conv", "prompt": "User: पौधे अपना भोजन कैसे बनाते हैं?\nAssistant:", "keywords": ["प्रकाश संश्लेषण", "सूर्य", "भोजन", "पौधे", "पत्तियां"]},
    {"id": "hi_27", "category": "hindi_conv", "prompt": "User: भारत की सबसे लंबी नदी कौन सी है?\nAssistant:", "keywords": ["गंगा", "नदी", "लंबी", "भारत"]},
    {"id": "hi_28", "category": "hindi_conv", "prompt": "User: सूर्य से हमें कौन सा विटामिन मिलता है?\nAssistant:", "keywords": ["विटामिन डी", "धूप", "विटामिन", "सूर्य"]},
    {"id": "hi_29", "category": "hindi_conv", "prompt": "User: हमारे शरीर में पानी का क्या महत्व है?\nAssistant:", "keywords": ["जल", "पानी", "शरीर", "स्वास्थ्य", "आवश्यक"]},
    {"id": "hi_30", "category": "hindi_conv", "prompt": "User: न्यूटन ने गति के कितने नियम दिए?\nAssistant:", "keywords": ["तीन", "नियम", "न्यूटन", "गति"]},

    # 5 Hinglish Conv
    {"id": "hg_26", "category": "hinglish_conv", "prompt": "User: Database mein primary key ka kya role hai?\nAssistant:", "keywords": ["unique", "record", "table", "key", "identify"]},
    {"id": "hg_27", "category": "hinglish_conv", "prompt": "User: Git commit aur push mein kya farak hota hai?\nAssistant:", "keywords": ["local", "remote", "commit", "push", "repository"]},
    {"id": "hg_28", "category": "hinglish_conv", "prompt": "User: Frontend development mein HTML aur CSS ka role kya hai?\nAssistant:", "keywords": ["structure", "style", "ui", "page", "web"]},
    {"id": "hg_29", "category": "hinglish_conv", "prompt": "User: Recursion programming technique kya hoti hai?\nAssistant:", "keywords": ["function", "calls", "itself", "base", "case"]},
    {"id": "hg_30", "category": "hinglish_conv", "prompt": "User: SQL aur NoSQL database mein kya difference hai?\nAssistant:", "keywords": ["relational", "schema", "table", "structure", "data"]},

    # 5 Instruction Following
    {"id": "inst_21", "category": "instruction_following", "prompt": "User: Answer with only the digit 7.\nAssistant:", "constraint": "exact_phrase", "expected": "7"},
    {"id": "inst_22", "category": "instruction_following", "prompt": "User: Reply with only 'AFFIRMATIVE'.\nAssistant:", "constraint": "exact_phrase", "expected": "affirmative"},
    {"id": "inst_23", "category": "instruction_following", "prompt": "User: Answer in one word: What is 2 plus 2?\nAssistant:", "constraint": "one_word", "expected": "4"},
    {"id": "inst_24", "category": "instruction_following", "prompt": "User: Reply with only 'YES' or 'NO': Can birds fly?\nAssistant:", "constraint": "yes_no", "expected": "yes"},
    {"id": "inst_25", "category": "instruction_following", "prompt": "User: Output only the string 'COMPLETE'.\nAssistant:", "constraint": "exact_phrase", "expected": "complete"},

    # 5 Multi-turn Context
    {"id": "mem_16", "category": "multi_turn", "history": ["User: My favorite fruit is mango.\nAssistant: Mango noted."], "prompt": "User: What is my favorite fruit?\nAssistant:", "expected": "mango"},
    {"id": "mem_17", "category": "multi_turn", "history": ["User: My father's name is Ramesh.\nAssistant: Hello Ramesh."], "prompt": "User: What is my father's name?\nAssistant:", "expected": "Ramesh"},
    {"id": "mem_18", "category": "multi_turn", "history": ["User: I live in Hyderabad.\nAssistant: Hyderabad noted."], "prompt": "User: What city do I live in?\nAssistant:", "expected": "Hyderabad"},
    {"id": "mem_19", "category": "multi_turn", "history": ["User: My cat is named Luna.\nAssistant: Luna is a nice name."], "prompt": "User: What is my cat's name?\nAssistant:", "expected": "Luna"},
    {"id": "mem_20", "category": "multi_turn", "history": ["User: I speak Hindi.\nAssistant: Bahut accha."], "prompt": "User: What language do I speak?\nAssistant:", "expected": "Hindi"},
]

total_260 = benchmark_items + fresh_tasks
out_dir = Path("data/sealed_eval_014")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "sealed_benchmark_260.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(total_260, f, indent=2, ensure_ascii=False)

print(f"Created fresh sealed benchmark with {len(total_260)} tasks at: {out_file}")
