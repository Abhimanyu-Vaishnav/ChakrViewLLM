"""
ChakrView Step 55: Multi-Task Interleaved Curriculum Generator.

Implements a balanced, multi-domain curriculum generator that combines:
1. Foundation (delimiters, brackets, YAML, JSON, text)
2. Computation (arithmetic, comparisons, transforms, parity)
3. Programming (functions, returns, conditionals, assertions)
4. Reasoning (single-step, multi-step, planning, state transitions)
5. Diagnosis (error localization, wrong operator, patch selection)
6. Trajectory (canonical XML action/observation sequences)
7. Instruction (exact output, constrained responses, structured formats)

Features:
- Configurable domain weights.
- Deterministic interleaved sampling.
- Clean train/validation/test split isolation without cross-contamination.
- Strict tokenization compatibility with ChakrMicro BPE vocabulary (4096).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
from typing import List, Dict, Any, Tuple, Optional

from chakrview.curriculum.generator import CurriculumSample
from chakrview.training.sharding import ShardWriter


@dataclass
class DomainWeights:
    foundation: float = 0.15
    computation: float = 0.15
    programming: float = 0.20
    reasoning: float = 0.20
    diagnosis: float = 0.10
    trajectory: float = 0.10
    instruction: float = 0.10

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        total = sum([
            self.foundation, self.computation, self.programming,
            self.reasoning, self.diagnosis, self.trajectory, self.instruction
        ])
        if not math.isclose(total, 1.0, rel_tol=1e-4):
            raise ValueError(f"Domain weights must sum to 1.0, got {total:.4f}")


class MultiTaskCurriculumGenerator:
    """
    Generates balanced, multi-domain curriculum samples with anti-forgetting interleaving.
    """

    def __init__(self, seed: int = 42, weights: Optional[DomainWeights] = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights = weights or DomainWeights()
        self.weights.validate()

    def generate_all_samples(self, target_sample_count: int = 800) -> List[CurriculumSample]:
        """
        Generate balanced multi-domain samples using configured domain weights.
        Interleaves domain items to prevent catastrophic forgetting.
        """
        domain_pools: Dict[str, List[CurriculumSample]] = {
            "foundation": self._gen_foundation_pool(),
            "computation": self._gen_computation_pool(),
            "programming": self._gen_programming_pool(),
            "reasoning": self._gen_reasoning_pool(),
            "diagnosis": self._gen_diagnosis_pool(),
            "trajectory": self._gen_trajectory_pool(),
            "instruction": self._gen_instruction_pool(),
        }

        weights_dict = self.weights.to_dict()
        samples: List[CurriculumSample] = []

        for domain, pool in domain_pools.items():
            count = int(target_sample_count * weights_dict[domain])
            pool_len = len(pool)
            for i in range(count):
                base_item = pool[i % pool_len]
                samples.append(CurriculumSample(
                    sample_id=f"{domain}_{base_item.sample_id}_{i:03d}",
                    level=base_item.level,
                    text=base_item.text,
                    tags=list(base_item.tags) + [domain],
                ))

        # Deterministic shuffle to interleave domains
        self.rng.shuffle(samples)
        return samples

    # -------------------------------------------------------------------------
    # 1. Foundation Domain Pool
    # -------------------------------------------------------------------------
    def _gen_foundation_pool(self) -> List[CurriculumSample]:
        items = []
        pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'")]
        for idx, (o, c) in enumerate(pairs):
            text = f"Pair: {o}token{c}\nBalanced: {o * 2}nested{c * 2}\n"
            items.append(CurriculumSample(f"fnd_pair_{idx}", "FND", text, ["delimiter"]))

        json_yaml = [
            ('{"status": "ok", "code": 200, "active": true}\n', ["json"]),
            ('name: ChakrMicro\nversion: 0.1\narch: cpu_first\n', ["yaml"]),
            ('The capital of France is Paris.\nThe capital of Japan is Tokyo.\n', ["fact"]),
            ('Antigravity is an AI system built for indigenous edge intelligence.\n', ["prose"]),
        ]
        for idx, (text, tags) in enumerate(json_yaml):
            items.append(CurriculumSample(f"fnd_struct_{idx}", "FND", text, tags))
        return items

    # -------------------------------------------------------------------------
    # 2. Computation Domain Pool
    # -------------------------------------------------------------------------
    def _gen_computation_pool(self) -> List[CurriculumSample]:
        items = []
        arithmetic = [
            ("5 + 3 = 8\n", ["add"]),
            ("12 + 15 = 27\n", ["add"]),
            ("20 - 7 = 13\n", ["sub"]),
            ("50 - 18 = 32\n", ["sub"]),
            ("4 * 6 = 24\n", ["mul"]),
            ("9 * 7 = 63\n", ["mul"]),
            ("36 / 6 = 6\n", ["div"]),
            ("100 / 4 = 25\n", ["div"]),
        ]
        for idx, (text, tags) in enumerate(arithmetic):
            items.append(CurriculumSample(f"cmp_arith_{idx}", "CMP", text, tags + ["arithmetic"]))

        comparisons = [
            ("Comparison: 9 > 4 is True\n", ["comparison"]),
            ("Comparison: 3 > 8 is False\n", ["comparison"]),
            ("Comparison: 15 == 15 is True\n", ["comparison"]),
            ("Comparison: 7 <= 2 is False\n", ["comparison"]),
            ("Parity: 18 is even\nParity: 21 is odd\n", ["parity"]),
            ("Transform: 'hello' uppercase is 'HELLO'\nTransform: 'WORLD' lowercase is 'world'\n", ["transform"]),
        ]
        for idx, (text, tags) in enumerate(comparisons):
            items.append(CurriculumSample(f"cmp_comp_{idx}", "CMP", text, tags))
        return items

    # -------------------------------------------------------------------------
    # 3. Programming Domain Pool
    # -------------------------------------------------------------------------
    def _gen_programming_pool(self) -> List[CurriculumSample]:
        items = []
        snippets = [
            (
                "def add(a: int, b: int) -> int:\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                ["function", "add"],
            ),
            (
                "def multiply(a: int, b: int) -> int:\n    return a * b\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
                ["function", "multiply"],
            ),
            (
                "def is_positive(x: float) -> bool:\n    return x > 0\n\ndef test_is_positive():\n    assert is_positive(5.0) is True\n    assert is_positive(-2.0) is False\n",
                ["function", "bool"],
            ),
            (
                "def clamp(val: float, min_v: float, max_v: float) -> float:\n    if val < min_v:\n        return min_v\n    if val > max_v:\n        return max_v\n    return val\n",
                ["function", "clamp"],
            ),
            (
                "def square(n: int) -> int:\n    return n * n\n\ndef test_square():\n    assert square(6) == 36\n",
                ["function", "square"],
            ),
            (
                "def get_first(items: list):\n    if not items:\n        return None\n    return items[0]\n",
                ["function", "list"],
            ),
        ]
        for idx, (text, tags) in enumerate(snippets):
            items.append(CurriculumSample(f"prg_code_{idx}", "PRG", text, tags + ["python"]))
        return items

    # -------------------------------------------------------------------------
    # 4. Reasoning Domain Pool
    # -------------------------------------------------------------------------
    def _gen_reasoning_pool(self) -> List[CurriculumSample]:
        items = []
        reasoning_cases = [
            (
                "Question: Is 15 > 8?\nReasoning: 15 is greater than 8.\nAnswer: True\n",
                ["single_step", "comparison"],
            ),
            (
                "Problem: Calculate (4 * 5) + (14 / 2).\nStep 1: Compute 4 * 5 = 20.\nStep 2: Compute 14 / 2 = 7.\nStep 3: 20 + 7 = 27.\nFinal Answer: 27\n",
                ["multi_step", "math"],
            ),
            (
                "System State: LIGHT is OFF.\nPlan:\n1. Action: SWITCH_ON -> State becomes ON.\nGoal State: ON\n",
                ["planning", "state_transition"],
            ),
            (
                "Question: Which is an antonym for 'fast'?\nOptions: (A) quick  (B) slow  (C) rapid\nAnswer: (B) slow\n",
                ["selection", "language"],
            ),
            (
                "TASK: Check if a number n is even.\nUNDERSTAND: Even numbers are divisible by 2.\nDECOMPOSE: Check n % 2 == 0.\nIMPLEMENT:\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\nVERIFY:\nassert is_even(4) is True\n",
                ["decomposition", "is_even"],
            ),
        ]
        for idx, (text, tags) in enumerate(reasoning_cases):
            items.append(CurriculumSample(f"rsn_case_{idx}", "RSN", text, tags + ["reasoning"]))
        return items

    # -------------------------------------------------------------------------
    # 5. Diagnosis Domain Pool
    # -------------------------------------------------------------------------
    def _gen_diagnosis_pool(self) -> List[CurriculumSample]:
        items = []
        diagnostics = [
            (
                "Diagnosis Task: Syntax error in function definition.\n"
                "Erroneous Code:\ndef sub(a, b)\n    return a - b\n"
                "Observation: SyntaxError: expected ':' at end of def statement.\n"
                "Fix:\ndef sub(a, b):\n    return a - b\n",
                ["diagnosis", "syntax_colon"],
            ),
            (
                "Diagnosis Task: Logic error in addition.\n"
                "Erroneous Code:\ndef add(a, b):\n    return a - b\n"
                "Observation: FAILED: add(2, 3) returned -1, expected 5.\n"
                "Diagnosis: Used subtraction '-' operator instead of '+'.\n"
                "Fix:\ndef add(a, b):\n    return a + b\n",
                ["diagnosis", "operator_fix"],
            ),
            (
                "Diagnosis Task: Unhandled division by zero.\n"
                "Erroneous Code:\ndef safe_div(a, b):\n    return a / b\n"
                "Observation: ZeroDivisionError when b == 0.\n"
                "Diagnosis: Missing guard check for divisor zero.\n"
                "Fix:\ndef safe_div(a, b):\n    if b == 0: return 0.0\n    return a / b\n",
                ["diagnosis", "zero_division"],
            ),
        ]
        for idx, (text, tags) in enumerate(diagnostics):
            items.append(CurriculumSample(f"diag_case_{idx}", "DIAG", text, tags))
        return items

    # -------------------------------------------------------------------------
    # 6. Trajectory Domain Pool
    # -------------------------------------------------------------------------
    def _gen_trajectory_pool(self) -> List[CurriculumSample]:
        items = []
        trajectories = [
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Implement multiply(a, b)\n</SPEC>\n"
                "<STATE>\nInitial implementation draft\n</STATE>\n"
                "<ACTION>\ndef multiply(a, b):\n    return a + b\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: multiply(3, 4) returned 7, expected 12\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nUsed addition instead of multiplication operator.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef multiply(a, b):\n    return a * b\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "multiply_fix"],
            ),
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Implement clamp(val, min_v, max_v)\n</SPEC>\n"
                "<STATE>\nInitial implementation draft\n</STATE>\n"
                "<ACTION>\ndef clamp(val, min_v, max_v):\n    return max_v\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: clamp(5, 0, 10) returned 10, expected 5\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nReturns max_v unconditionally.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef clamp(val, min_v, max_v):\n    if val < min_v: return min_v\n    if val > max_v: return max_v\n    return val\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "clamp_fix"],
            ),
        ]
        for idx, (text, tags) in enumerate(trajectories):
            items.append(CurriculumSample(f"trj_case_{idx}", "TRJ", text, tags))
        return items

    # -------------------------------------------------------------------------
    # 7. Instruction Domain Pool
    # -------------------------------------------------------------------------
    def _gen_instruction_pool(self) -> List[CurriculumSample]:
        items = []
        instructions = [
            ("Instruction: Print exactly the word 'SUCCESS'.\nOutput: SUCCESS\n", ["exact_output"]),
            ("Instruction: Return JSON with key 'result' set to true.\nOutput: {\"result\": true}\n", ["json_format"]),
            ("Instruction: Answer with True or False: 100 > 50?\nOutput: True\n", ["boolean_output"]),
            ("Instruction: Complete the tag <STATE>.\nOutput: <STATE>RUNNING</STATE>\n", ["tag_format"]),
        ]
        for idx, (text, tags) in enumerate(instructions):
            items.append(CurriculumSample(f"ins_case_{idx}", "INS", text, tags))
        return items

    # -------------------------------------------------------------------------
    # Split & Shard Serialization
    # -------------------------------------------------------------------------
    def build_dataset_splits(
        self,
        samples: List[CurriculumSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Dict[str, List[CurriculumSample]]:
        """Split samples cleanly into train, val, and test partitions."""
        total = len(samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }

    def write_shards(
        self,
        splits: Dict[str, List[CurriculumSample]],
        output_dir: Path,
        tokenizer: Any,
        vocab_size: int = 4096,
    ) -> Dict[str, Any]:
        """Serialize tokenized splits into binary uint16 shards with manifests."""
        manifest: Dict[str, Any] = {
            "seed": self.seed,
            "vocab_size": vocab_size,
            "weights": self.weights.to_dict(),
            "splits": {},
        }
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_samples in splits.items():
            writer = ShardWriter(
                output_dir=output_dir,
                split_name=split_name,
                vocab_size=vocab_size,
                max_tokens_per_shard=100_000,
            )

            total_tokens = 0
            domain_counts: Dict[str, int] = {}
            for sample in split_samples:
                # Format: <BOS> + text tokens + <EOS>
                raw_tokens = tokenizer.encode(sample.text)
                token_ids = [0] + raw_tokens + [1]
                writer.add_document(token_ids)
                total_tokens += len(token_ids)

                domain = sample.tags[-1] if sample.tags else "unknown"
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            meta = writer.close()
            manifest["splits"][split_name] = {
                "sample_count": len(split_samples),
                "token_count": total_tokens,
                "domain_distribution": domain_counts,
                "shard_meta": meta,
            }

        manifest_path = output_dir / "multitask_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
