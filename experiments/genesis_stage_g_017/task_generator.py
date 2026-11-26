"""
NR-017 Phase 1 & 2: Verifiable Algorithmic Task Generator & State-Trace Formatter
Generates deterministic tasks across 10 algorithmic families:
1. variable_binding
2. variable_renaming
3. arithmetic_operations
4. sequential_state_updates
5. conditional_branching
6. nested_operations
7. list_transformations
8. string_transformations
9. stack_operations
10. algorithmic_compositions

Each sample contains:
- task_id, family, difficulty (Level 1, Level 2, Level 3)
- input prompt: <task> ... </task>
- Condition A target (Direct Answer): <answer> ... </answer>
- Condition B target (Explicit State Trace): <state_1> ... </state_1> <state_2> ... <answer> ... </answer>
- Condition C target (Structured Compact Trace): step1 -> step2 -> answer=...
- intermediate_states: list of expected verifiable states
- ground_truth_answer: final answer
- is_ood: boolean
"""

import sys
import json
import random
import hashlib
from pathlib import Path

class VerifiableTaskGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Disjoint Variable Sets
        self.in_vars = ['x', 'y', 'z', 'a', 'b', 'c', 'p', 'q', 'm', 'n']
        self.ood_vars = ['alpha', 'beta', 'gamma', 'delta', 'omega', 'sigma', 'theta', 'phi']

        # Disjoint Numeric Ranges
        self.in_range_l1 = (2, 20)
        self.in_range_l2 = (10, 50)
        self.in_range_l3 = (20, 80)

        self.ood_range = (85, 160)

    def generate_task(self, family: str, difficulty: int = 1, is_ood: bool = False, task_idx: int = 1):
        vars_pool = self.ood_vars if is_ood else self.in_vars
        val_range = self.ood_range if is_ood else (
            self.in_range_l1 if difficulty == 1 else (self.in_range_l2 if difficulty == 2 else self.in_range_l3)
        )

        task_desc = ""
        states = []
        answer = ""

        # 1. Variable Binding
        if family == "variable_binding":
            v1, v2 = self.rng.sample(vars_pool, 2)
            n1 = self.rng.randint(*val_range)
            n2 = self.rng.randint(*val_range)
            task_desc = f"Bind {v1} = {n1} and {v2} = {n2}. Compute {v1} + {v2}."
            s1 = f"{v1}={n1}"
            s2 = f"{v2}={n2}"
            s3 = f"sum={n1 + n2}"
            states = [s1, s2, s3]
            answer = str(n1 + n2)

        # 2. Variable Renaming
        elif family == "variable_renaming":
            v1, v2 = self.rng.sample(vars_pool, 2)
            n1 = self.rng.randint(*val_range)
            task_desc = f"Set {v1} = {n1}. Let {v2} = {v1} + 10. Find {v2}."
            s1 = f"{v1}={n1}"
            s2 = f"{v2}={n1}+10={n1+10}"
            states = [s1, s2]
            answer = str(n1 + 10)

        # 3. Arithmetic Operations
        elif family == "arithmetic_operations":
            n1 = self.rng.randint(*val_range)
            n2 = self.rng.randint(2, 9)
            task_desc = f"Compute {n1} * {n2}."
            s1 = f"mult={n1}*{n2}={n1*n2}"
            states = [s1]
            answer = str(n1 * n2)

        # 4. Sequential State Updates
        elif family == "sequential_state_updates":
            start = self.rng.randint(*val_range)
            d1 = self.rng.randint(3, 15)
            d2 = self.rng.randint(2, 8)
            task_desc = f"Initial state S = {start}. Add {d1}, then subtract {d2}."
            s1 = f"S0={start}"
            s2 = f"S1={start}+{d1}={start+d1}"
            s3 = f"S2={start+d1}-{d2}={start+d1-d2}"
            states = [s1, s2, s3]
            answer = str(start + d1 - d2)

        # 5. Conditional Branching
        elif family == "conditional_branching":
            threshold = 50 if not is_ood else 100
            n = self.rng.randint(*val_range)
            task_desc = f"If value {n} > {threshold}, output YES and multiply by 2; else output NO and add 5."
            if n > threshold:
                s1 = f"eval={n}>{threshold}:TRUE"
                s2 = f"branch=YES,result={n*2}"
                ans_str = f"YES {n*2}"
            else:
                s1 = f"eval={n}>{threshold}:FALSE"
                s2 = f"branch=NO,result={n+5}"
                ans_str = f"NO {n+5}"
            states = [s1, s2]
            answer = ans_str

        # 6. Nested Operations
        elif family == "nested_operations":
            a = self.rng.randint(2, 10 if not is_ood else 25)
            b = self.rng.randint(2, 9)
            c = self.rng.randint(1, 15)
            task_desc = f"Evaluate ({a} * {b}) + {c}."
            s1 = f"inner={a}*{b}={a*b}"
            s2 = f"outer={a*b}+{c}={a*b+c}"
            states = [s1, s2]
            answer = str(a * b + c)

        # 7. List Transformations
        elif family == "list_transformations":
            lst = [self.rng.randint(1, 30 if not is_ood else 70) for _ in range(4)]
            task_desc = f"Take list {lst}. Step 1: reverse list. Step 2: find maximum."
            rev = list(reversed(lst))
            max_v = max(lst)
            s1 = f"reversed={rev}"
            s2 = f"max={max_v}"
            states = [s1, s2]
            answer = str(max_v)

        # 8. String Transformations
        elif family == "string_transformations":
            words = ['tensor', 'matrix', 'stream', 'token'] if not is_ood else ['gradient', 'vector', 'chakra', 'neural']
            w = self.rng.choice(words)
            task_desc = f"Take word '{w}'. Step 1: uppercase. Step 2: reverse."
            up = w.upper()
            rev = up[::-1]
            s1 = f"upper={up}"
            s2 = f"reversed={rev}"
            states = [s1, s2]
            answer = rev

        # 9. Stack Operations
        elif family == "stack_operations":
            n1 = self.rng.randint(1, 20)
            n2 = self.rng.randint(21, 40)
            n3 = self.rng.randint(41, 60)
            task_desc = f"Stack operations: PUSH {n1}, PUSH {n2}, POP, PUSH {n3}. What is the top element?"
            s1 = f"stack=[{n1}]"
            s2 = f"stack=[{n1},{n2}]"
            s3 = f"pop={n2},stack=[{n1}]"
            s4 = f"stack=[{n1},{n3}]"
            states = [s1, s2, s3, s4]
            answer = str(n3)

        # 10. Algorithmic Compositions
        else: # algorithmic_compositions
            v = self.rng.choice(vars_pool)
            val = self.rng.randint(*val_range)
            task_desc = f"Variable {v} = {val}. Double it, then check if result is even: answer True or False."
            res = val * 2
            is_even = (res % 2 == 0)
            s1 = f"{v}={val}"
            s2 = f"double={val}*2={res}"
            s3 = f"is_even={res}%2==0:{is_even}"
            states = [s1, s2, s3]
            answer = str(is_even)

        # Format Condition Targets
        prompt = f"<task> {task_desc} </task>"
        # Condition A: Direct Answer
        target_a = f"<answer> {answer} </answer>"
        # Condition B: Explicit State Trace
        state_tags = "".join([f"<state_{i+1}> {st} </state_{i+1}> " for i, st in enumerate(states)])
        target_b = f"{state_tags}<answer> {answer} </answer>"
        # Condition C: Structured Compact Trace
        compact_trace = " -> ".join(states) + f" -> answer={answer}"
        target_c = f"<trace> {compact_trace} </trace>"

        return {
            "task_id": f"NR17-{task_idx:04d}",
            "family": family,
            "difficulty": difficulty,
            "is_ood": is_ood,
            "prompt": prompt,
            "states": states,
            "ground_truth_answer": answer,
            "target_a": target_a,
            "target_b": target_b,
            "target_c": target_c
        }

    def generate_dataset_suite(self, train_count=1800, sealed_test_count=400):
        families = [
            "variable_binding",
            "variable_renaming",
            "arithmetic_operations",
            "sequential_state_updates",
            "conditional_branching",
            "nested_operations",
            "list_transformations",
            "string_transformations",
            "stack_operations",
            "algorithmic_compositions"
        ]

        train_tasks = []
        for i in range(train_count):
            fam = families[i % len(families)]
            diff = (i % 3) + 1 # Level 1, 2, 3
            task = self.generate_task(fam, difficulty=diff, is_ood=False, task_idx=i+1)
            train_tasks.append(task)

        test_tasks = []
        for j in range(sealed_test_count):
            fam = families[j % len(families)]
            diff = (j % 3) + 1
            # 50% in-distribution, 50% strict OOD
            is_ood = (j >= sealed_test_count // 2)
            task = self.generate_task(fam, difficulty=diff, is_ood=is_ood, task_idx=train_count + j + 1)
            test_tasks.append(task)

        return train_tasks, test_tasks

if __name__ == '__main__':
    gen = VerifiableTaskGenerator(seed=777)
    train_tasks, test_tasks = gen.generate_dataset_suite(train_count=1800, sealed_test_count=400)

    out_dir = Path("data/stage_g_017")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "stage_g_train_1800.json"
    test_file = out_dir / "stage_g_sealed_benchmark_400.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_tasks, f, indent=2)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_tasks, f, indent=2)

    def hash_file(p):
        with open(p, 'rb') as fp: return hashlib.sha256(fp.read()).hexdigest()

    print(f"Train tasks generated: {len(train_tasks)} | SHA-256: {hash_file(train_file)}")
    print(f"Sealed benchmark tasks generated: {len(test_tasks)} | SHA-256: {hash_file(test_file)}")
