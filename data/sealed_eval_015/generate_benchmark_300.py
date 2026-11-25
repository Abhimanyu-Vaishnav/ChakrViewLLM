"""
NR-015 Phase 2: Controlled Generalization Benchmark Generator
Generates exactly 300 unseen tasks across 12 distinct task families:
1. simple_arithmetic (25 tasks)
2. sequence_continuation (25 tasks)
3. variable_binding (25 tasks)
4. function_argument_binding (25 tasks)
5. io_transformation (25 tasks)
6. conditional_logic (25 tasks)
7. list_transformation (25 tasks)
8. string_transformation (25 tasks)
9. rule_induction (25 tasks)
10. python_function_generation (40 tasks with executable hidden tests)
11. multistep_instruction (30 tasks)
12. novel_combinations (30 tasks)
Total: 300 tasks
"""

import json
import random
from pathlib import Path

def generate_benchmark_300(seed=42):
    random.seed(seed)
    tasks = []
    task_id = 1

    # 1. Simple Arithmetic Patterns (25 tasks)
    for i in range(25):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            ans = str(a + b)
            prompt = f"Calculate {a} + {b}."
        elif op == '-':
            ans = str(a - b)
            prompt = f"Calculate {a} - {b}."
        else:
            a_s = random.randint(3, 12)
            b_s = random.randint(3, 12)
            ans = str(a_s * b_s)
            prompt = f"Calculate {a_s} * {b_s}."
        tasks.append({
            "task_id": f"GEN-01-{task_id:03d}",
            "family": "simple_arithmetic",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 2. Sequence Continuation (25 tasks)
    for i in range(25):
        start = random.randint(2, 20)
        step = random.randint(2, 7)
        seq = [start + j * step for j in range(5)]
        next_val = start + 5 * step
        prompt = f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?"
        tasks.append({
            "task_id": f"GEN-02-{task_id:03d}",
            "family": "sequence_continuation",
            "prompt": prompt,
            "ground_truth": str(next_val),
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 3. Variable Binding (25 tasks)
    var_names = [('alpha', 'beta'), ('first_num', 'second_num'), ('delta', 'gamma'), ('foo', 'bar'), ('p', 'q')]
    for i in range(25):
        v1, v2 = random.choice(var_names)
        val1 = random.randint(5, 50)
        val2 = random.randint(5, 50)
        op = random.choice(['sum', 'difference', 'product'])
        if op == 'sum':
            ans = str(val1 + val2)
            prompt = f"Let {v1} = {val1} and {v2} = {val2}. What is the sum of {v1} and {v2}?"
        elif op == 'difference':
            ans = str(val1 - val2)
            prompt = f"Let {v1} = {val1} and {v2} = {val2}. What is {v1} minus {v2}?"
        else:
            val1_s = random.randint(2, 9)
            val2_s = random.randint(2, 9)
            ans = str(val1_s * val2_s)
            prompt = f"Let {v1} = {val1_s} and {v2} = {val2_s}. What is {v1} multiplied by {v2}?"
        tasks.append({
            "task_id": f"GEN-03-{task_id:03d}",
            "family": "variable_binding",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 4. Function Argument Binding (25 tasks)
    funcs = [
        ("compute_area(w, h)", "w * h", lambda w, h: w * h, "width w and height h"),
        ("perimeter(length, width)", "2 * (length + width)", lambda l, w: 2 * (l + w), "length and width"),
        ("linear_func(x, m, c)", "m * x + c", lambda x, m, c: m * x + c, "input x, slope m, intercept c"),
        ("combine_ratio(a, b)", "a / (a + b)", lambda a, b: f"{(a / (a + b)):.2f}", "quantities a and b")
    ]
    for i in range(25):
        w = random.randint(3, 15)
        h = random.randint(3, 15)
        ans = str(w * h)
        prompt = f"Given a function calculate_area(w, h) = w * h, evaluate calculate_area({w}, {h})."
        tasks.append({
            "task_id": f"GEN-04-{task_id:03d}",
            "family": "function_argument_binding",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 5. Input-Output Transformation (25 tasks)
    for i in range(25):
        n = random.randint(2, 12)
        # rule: f(x) = 2*x + 1
        prompt = f"Input: 1 -> Output: 3\nInput: 2 -> Output: 5\nInput: 3 -> Output: 7\nWhat is the Output for Input: {n}?"
        ans = str(2 * n + 1)
        tasks.append({
            "task_id": f"GEN-05-{task_id:03d}",
            "family": "io_transformation",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 6. Conditional Logic (25 tasks)
    for i in range(25):
        threshold = random.randint(20, 80)
        val = random.randint(10, 90)
        ans = "PASS" if val >= threshold else "FAIL"
        prompt = f"Rule: If score is greater than or equal to {threshold}, output PASS. Otherwise output FAIL.\nGiven score = {val}, what is the output?"
        tasks.append({
            "task_id": f"GEN-06-{task_id:03d}",
            "family": "conditional_logic",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_match"
        })
        task_id += 1

    # 7. List Transformation (25 tasks)
    for i in range(25):
        lst = [random.randint(1, 30) for _ in range(4)]
        op = random.choice(['reverse', 'max', 'min', 'sum'])
        if op == 'reverse':
            ans = str(list(reversed(lst)))
            prompt = f"Reverse the list: {lst}."
        elif op == 'max':
            ans = str(max(lst))
            prompt = f"What is the maximum number in {lst}?"
        elif op == 'min':
            ans = str(min(lst))
            prompt = f"What is the minimum number in {lst}?"
        else:
            ans = str(sum(lst))
            prompt = f"What is the sum of the elements in {lst}?"
        tasks.append({
            "task_id": f"GEN-07-{task_id:03d}",
            "family": "list_transformation",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 8. String Transformation (25 tasks)
    words = ['python', 'neural', 'matrix', 'tensor', 'gradient', 'vector', 'chakra', 'genesis', 'stream', 'token']
    for i in range(25):
        w = random.choice(words)
        op = random.choice(['upper', 'reverse', 'length'])
        if op == 'upper':
            ans = w.upper()
            prompt = f"Convert the word '{w}' to uppercase."
        elif op == 'reverse':
            ans = w[::-1]
            prompt = f"Reverse the characters in the word '{w}'."
        else:
            ans = str(len(w))
            prompt = f"How many letters are in the word '{w}'?"
        tasks.append({
            "task_id": f"GEN-08-{task_id:03d}",
            "family": "string_transformation",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_match"
        })
        task_id += 1

    # 9. Rule Induction (25 tasks)
    for i in range(25):
        multiplier = random.choice([3, 4, 5, 10])
        val = random.randint(2, 9)
        prompt = f"Observed mappings:\n2 -> {2*multiplier}\n4 -> {4*multiplier}\n6 -> {6*multiplier}\nFollowing the same rule, what is the output for {val}?"
        ans = str(val * multiplier)
        tasks.append({
            "task_id": f"GEN-09-{task_id:03d}",
            "family": "rule_induction",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 10. Python Function Generation (40 tasks with executable tests)
    code_templates = [
        ("multiply_two(a, b)", "return a * b", "assert multiply_two(3, 4) == 12\nassert multiply_two(-2, 5) == -10"),
        ("is_even(n)", "return n % 2 == 0", "assert is_even(4) == True\nassert is_even(7) == False"),
        ("square_number(x)", "return x ** 2", "assert square_number(5) == 25\nassert square_number(-3) == 9"),
        ("first_element(lst)", "return lst[0] if lst else None", "assert first_element([10, 20]) == 10\nassert first_element([]) is None"),
        ("last_element(lst)", "return lst[-1] if lst else None", "assert last_element([1, 2, 3]) == 3\nassert last_element([]) is None"),
        ("string_length(s)", "return len(s)", "assert string_length('hello') == 5\nassert string_length('') == 0"),
        ("to_lowercase(s)", "return s.lower()", "assert to_lowercase('HELLO') == 'hello'\nassert to_lowercase('Py') == 'py'"),
        ("sum_list(lst)", "return sum(lst)", "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
        ("absolute_value(n)", "return abs(n)", "assert absolute_value(-5) == 5\nassert absolute_value(3) == 3"),
        ("find_maximum(a, b)", "return max(a, b)", "assert find_maximum(4, 9) == 9\nassert find_maximum(-1, -5) == -1")
    ]
    for i in range(40):
        sig, body, test = code_templates[i % len(code_templates)]
        fn_name = sig.split('(')[0]
        prompt = f"Write a Python function `{sig}` that performs the required operation.\nProvide only the executable python function definition."
        tasks.append({
            "task_id": f"GEN-10-{task_id:03d}",
            "family": "python_function_generation",
            "prompt": prompt,
            "function_name": fn_name,
            "test_assertions": test,
            "eval_type": "python_execution"
        })
        task_id += 1

    # 11. Multi-Step Instructions (30 tasks)
    for i in range(30):
        n = random.randint(5, 25)
        step1 = n * 2
        step2 = step1 + 5
        prompt = f"Step 1: Take the number {n} and double it.\nStep 2: Add 5 to the result from Step 1.\nWhat is the final number?"
        ans = str(step2)
        tasks.append({
            "task_id": f"GEN-11-{task_id:03d}",
            "family": "multistep_instruction",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_or_numeric"
        })
        task_id += 1

    # 12. Novel Combinations of Familiar Concepts (30 tasks)
    for i in range(30):
        w = random.choice(['code', 'data', 'byte', 'tree', 'flow'])
        times = random.randint(2, 4)
        ans = w.upper() * times
        prompt = f"Take the word '{w}', capitalize it to uppercase, and repeat it {times} times with no spaces."
        tasks.append({
            "task_id": f"GEN-12-{task_id:03d}",
            "family": "novel_combinations",
            "prompt": prompt,
            "ground_truth": ans,
            "eval_type": "exact_match"
        })
        task_id += 1

    print(f"Total benchmark tasks generated: {len(tasks)}")
    return tasks

if __name__ == '__main__':
    bench = generate_benchmark_300(seed=42)
    out_dir = Path("data/sealed_eval_015")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sealed_benchmark_300.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(bench, f, indent=2, ensure_ascii=False)
    print(f"Sealed benchmark saved to: {out_file}")
