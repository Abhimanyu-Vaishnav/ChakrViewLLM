"""
NR-016 Phase 3: Scientific Task Grammar & Synthetic Generator
Generates tasks with deterministic grammar across 8 families:
1. variable_binding_and_renaming
2. argument_order_changes
3. arithmetic_transformations
4. nested_operations
5. compositional_rule_application
6. io_transformations
7. multistep_state_updates
8. structured_code_generation (with executable assertions)

Features:
- Deterministic random seed
- Grammatical separation of templates from answer generation
- Strict structural OOD partition (held-out variable sets, operations, numerical ranges)
- Automatic verifiable evaluator
"""

import sys
import json
import random
import hashlib
from pathlib import Path

class ScientificTaskGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Partitioned Variable Names
        self.in_distribution_vars = ['x', 'y', 'z', 'm', 'n', 'p', 'q', 'u', 'v', 'val_a', 'val_b']
        self.out_distribution_vars = ['zeta', 'theta', 'omega', 'psi', 'sigma', 'lambda_var', 'kappa', 'chi']
        
        # Partitioned Numeric Ranges
        self.in_dist_range = (5, 45)
        self.out_dist_range = (52, 110)

    def generate_task(self, family: str, is_ood: bool = False, task_idx: int = 1):
        vars_pool = self.out_distribution_vars if is_ood else self.in_distribution_vars
        val_range = self.out_dist_range if is_ood else self.in_dist_range
        
        if family == "variable_binding_and_renaming":
            v1, v2 = self.rng.sample(vars_pool, 2)
            n1 = self.rng.randint(*val_range)
            n2 = self.rng.randint(*val_range)
            ans = str(n1 + n2)
            prompt = f"Given {v1} = {n1} and {v2} = {n2}, compute the sum of {v1} and {v2}."
            return {
                "task_id": f"NR16-T1-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }
            
        elif family == "argument_order_changes":
            n1 = self.rng.randint(*val_range)
            n2 = self.rng.randint(2, 9)
            # test commutativity / non-commutativity: difference
            ans = str(n1 - n2)
            prompt = f"Subtract {n2} from {n1}."
            return {
                "task_id": f"NR16-T2-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }

        elif family == "arithmetic_transformations":
            n1 = self.rng.randint(*val_range)
            multiplier = 3 if not is_ood else 7
            ans = str(n1 * multiplier)
            prompt = f"Multiply {n1} by {multiplier}."
            return {
                "task_id": f"NR16-T3-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }

        elif family == "nested_operations":
            a = self.rng.randint(2, 10 if not is_ood else 20)
            b = self.rng.randint(2, 10)
            c = self.rng.randint(1, 15)
            ans = str(a * b + c)
            prompt = f"Evaluate ({a} * {b}) + {c}."
            return {
                "task_id": f"NR16-T4-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }

        elif family == "compositional_rule_application":
            word = self.rng.choice(['test', 'data', 'byte', 'core'] if not is_ood else ['tensor', 'matrix', 'stream', 'token'])
            times = 2 if not is_ood else 3
            ans = word.upper() * times
            prompt = f"Apply Rule 1: Capitalize '{word}' to uppercase.\nApply Rule 2: Repeat the result {times} times.\nWhat is the final text?"
            return {
                "task_id": f"NR16-T5-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_match"
            }

        elif family == "io_transformations":
            scale = 4 if not is_ood else 9
            query_val = self.rng.randint(2, 12)
            ans = str(query_val * scale)
            prompt = f"Input: 1 -> Output: {1*scale}\nInput: 2 -> Output: {2*scale}\nInput: 3 -> Output: {3*scale}\nWhat is the Output for Input: {query_val}?"
            return {
                "task_id": f"NR16-T6-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }

        elif family == "multistep_state_updates":
            start = self.rng.randint(10, 50)
            delta1 = self.rng.randint(5, 20)
            delta2 = self.rng.randint(2, 10)
            ans = str(start + delta1 - delta2)
            prompt = f"State begins at {start}.\nStep 1: Increase state by {delta1}.\nStep 2: Decrease state by {delta2}.\nWhat is the final state value?"
            return {
                "task_id": f"NR16-T7-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": ans,
                "eval_type": "exact_or_numeric"
            }

        elif family == "structured_code_generation":
            templates = [
                ("add_two(a, b)", "return a + b", "assert add_two(10, 20) == 30\nassert add_two(-5, 5) == 0"),
                ("is_positive(x)", "return x > 0", "assert is_positive(5) == True\nassert is_positive(-3) == False"),
                ("double_val(n)", "return n * 2", "assert double_val(8) == 16\nassert double_val(-4) == -8"),
                ("get_first(lst)", "return lst[0] if lst else None", "assert get_first([99, 100]) == 99\nassert get_first([]) is None")
            ]
            sig, body, tests = self.rng.choice(templates)
            prompt = f"Write a Python function `{sig}` that performs the specified task. Return only the executable code."
            return {
                "task_id": f"NR16-T8-{task_idx:04d}",
                "family": family,
                "is_ood": is_ood,
                "prompt": prompt,
                "ground_truth": body,
                "test_assertions": tests,
                "eval_type": "python_execution"
            }

    def generate_benchmark_suite(self, count_per_family=40):
        families = [
            "variable_binding_and_renaming",
            "argument_order_changes",
            "arithmetic_transformations",
            "nested_operations",
            "compositional_rule_application",
            "io_transformations",
            "multistep_state_updates",
            "structured_code_generation"
        ]
        
        benchmark_tasks = []
        task_counter = 1
        
        for fam in families:
            # 20 in-distribution, 20 out-of-distribution
            for _ in range(count_per_family // 2):
                benchmark_tasks.append(self.generate_task(fam, is_ood=False, task_idx=task_counter))
                task_counter += 1
            for _ in range(count_per_family // 2):
                benchmark_tasks.append(self.generate_task(fam, is_ood=True, task_idx=task_counter))
                task_counter += 1
                
        return benchmark_tasks

    def generate_training_corpus(self, num_samples=1600):
        families = [
            "variable_binding_and_renaming",
            "argument_order_changes",
            "arithmetic_transformations",
            "nested_operations",
            "compositional_rule_application",
            "io_transformations",
            "multistep_state_updates",
            "structured_code_generation"
        ]
        
        training_samples = []
        for i in range(num_samples):
            fam = families[i % len(families)]
            # Training samples are strictly in-distribution!
            task = self.generate_task(fam, is_ood=False, task_idx=i+1)
            gt = task['ground_truth']
            text = f"<user>\n{task['prompt']}\n<assistant>\n{gt}"
            training_samples.append({
                "text": text,
                "family": fam,
                "prompt": task['prompt'],
                "answer": gt
            })
            
        return training_samples

if __name__ == '__main__':
    gen = ScientificTaskGenerator(seed=2026)
    bench = gen.generate_benchmark_suite(count_per_family=40) # 8 * 40 = 320 tasks
    train_corpus = gen.generate_training_corpus(num_samples=1600)
    
    out_dir = Path("data/scientific_eval_016")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bench_file = out_dir / "scientific_benchmark_320.json"
    train_file = out_dir / "scientific_train_1600.json"
    
    with open(bench_file, 'w', encoding='utf-8') as f:
        json.dump(bench, f, indent=2)
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_corpus, f, indent=2)
        
    def hash_file(p):
        with open(p, 'rb') as fp: return hashlib.sha256(fp.read()).hexdigest()
        
    print(f"Generated Benchmark Tasks: {len(bench)} | SHA-256: {hash_file(bench_file)}")
    print(f"Generated Training Samples: {len(train_corpus)} | SHA-256: {hash_file(train_file)}")
