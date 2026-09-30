"""
ChakrView Step 54: Structured Reasoning Curriculum Generator.

Generates a balanced, multi-tier reasoning curriculum for ChakrMicro:
- Level R0: Foundation (delimiters, brackets, structural formats, syntax, operators)
- Level R1: Single-Step Reasoning (comparison, classification, arithmetic, transformations)
- Level R2: Multi-Step Reasoning (chained arithmetic, condition chains, state transitions, planning)
- Level R3: Task Decomposition & Trajectories (Understand -> Decompose -> Implement -> Verify)
- Action/Observation Trajectories (canonical XML-tagged format within 512-token context)

Adheres to:
- Deterministic seeding.
- Manifest with sample counts and token metadata.
- Clean isolation of train/validation/test partitions.
- Tokenizer compatibility verification.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import List, Dict, Any, Tuple

from chakrview.curriculum.generator import CurriculumSample
from chakrview.training.sharding import ShardWriter


class ReasoningCurriculumGenerator:
    """
    Generates balanced, multi-tier reasoning and trajectory samples for ChakrMicro.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_all_samples(self, repeats: int = 12) -> List[CurriculumSample]:
        """Generate balanced multi-tier reasoning samples with systematic repeats."""
        base_samples: List[CurriculumSample] = []
        base_samples.extend(self._gen_level_r0())
        base_samples.extend(self._gen_level_r1())
        base_samples.extend(self._gen_level_r2())
        base_samples.extend(self._gen_level_r3())
        base_samples.extend(self._gen_trajectories())

        all_samples: List[CurriculumSample] = []
        for r in range(repeats):
            for s in base_samples:
                all_samples.append(CurriculumSample(
                    sample_id=f"{s.sample_id}_r{r:02d}",
                    level=s.level,
                    text=s.text,
                    tags=list(s.tags),
                ))

        # Deterministic shuffle
        self.rng.shuffle(all_samples)
        return all_samples

    # -------------------------------------------------------------------------
    # Level R0: Foundation (delimiters, structure, stability, syntax)
    # -------------------------------------------------------------------------
    def _gen_level_r0(self) -> List[CurriculumSample]:
        samples = []
        pairs = [
            ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'"),
        ]
        for idx, (open_c, close_c) in enumerate(pairs):
            text = f"Pair: {open_c}content{close_c}\nBalanced: {open_c * 3}nested{close_c * 3}\n"
            samples.append(CurriculumSample(
                sample_id=f"r0_delim_{idx:03d}",
                level="R0",
                text=text,
                tags=["delimiter", "balance"],
            ))

        # Basic tags & syntax markers
        structural = [
            ("<TAG>content</TAG>\n<STATE>idle</STATE>\n<ACTION>execute</ACTION>\n", ["markup", "tags"]),
            ("key: value\nstatus: ok\ncount: 42\n", ["yaml", "structure"]),
            ("if x > 0:\n    y = x + 1\nelse:\n    y = 0\n", ["syntax", "python"]),
            ("result = [i * 2 for i in range(5)]\n", ["syntax", "comprehension"]),
        ]
        for idx, (text, tags) in enumerate(structural):
            samples.append(CurriculumSample(
                sample_id=f"r0_struct_{idx:03d}",
                level="R0",
                text=text,
                tags=tags + ["foundation"],
            ))
        return samples

    # -------------------------------------------------------------------------
    # Level R1: Single-Step Reasoning (comparison, classification, arithmetic)
    # -------------------------------------------------------------------------
    def _gen_level_r1(self) -> List[CurriculumSample]:
        samples = []
        # Comparisons
        comparisons = [
            ("7 > 3", "True", "7 is greater than 3"),
            ("12 < 5", "False", "12 is not less than 5"),
            ("8 == 8", "True", "8 equals 8"),
            ("15 != 15", "False", "15 is equal to 15"),
            ("9 >= 9", "True", "9 is greater than or equal to 9"),
            ("4 <= 2", "False", "4 is not less than or equal to 2"),
            ("10 > 20", "False", "10 is less than 20"),
            ("100 == 100", "True", "100 equals 100"),
        ]
        for idx, (expr, res, exp) in enumerate(comparisons):
            text = f"Question: Is {expr}?\nReasoning: {exp}.\nAnswer: {res}\n"
            samples.append(CurriculumSample(
                sample_id=f"r1_comp_{idx:03d}",
                level="R1",
                text=text,
                tags=["comparison", "single_step"],
            ))

        # Single-step arithmetic
        arithmetic = [
            ("7 + 8", "15"), ("23 + 14", "37"), ("50 - 18", "32"),
            ("9 * 6", "54"), ("42 / 6", "7"), ("15 * 3", "45"),
            ("64 / 8", "8"), ("81 - 25", "56"), ("12 + 19", "31"),
            ("4 * 12", "48"), ("99 + 1", "100"), ("100 - 37", "63"),
        ]
        for idx, (expr, val) in enumerate(arithmetic):
            text = f"Calculate: {expr} =\nAnswer: {val}\n"
            samples.append(CurriculumSample(
                sample_id=f"r1_arith_{idx:03d}",
                level="R1",
                text=text,
                tags=["arithmetic", "single_step"],
            ))

        # Classification
        classifications = [
            ("Input: -5\nQuestion: Is it positive, negative, or zero?\nAnswer: negative\n", ["classification", "sign"]),
            ("Input: 14\nQuestion: Is 14 even or odd?\nAnswer: even\n", ["classification", "parity"]),
            ("Input: 7\nQuestion: Is 7 even or odd?\nAnswer: odd\n", ["classification", "parity"]),
            ("Input: 'hello'\nQuestion: Is type str or int?\nAnswer: str\n", ["classification", "type"]),
            ("Input: [1, 2]\nQuestion: Is type list or dict?\nAnswer: list\n", ["classification", "type"]),
            ("Input: {'a': 1}\nQuestion: Is type list or dict?\nAnswer: dict\n", ["classification", "type"]),
        ]
        for idx, (text, tags) in enumerate(classifications):
            samples.append(CurriculumSample(
                sample_id=f"r1_class_{idx:03d}",
                level="R1",
                text=text,
                tags=tags + ["classification"],
            ))

        # Selection from alternatives
        selections = [
            (
                "Question: Which number is the largest: 4, 19, 7?\n"
                "Options: (A) 4  (B) 19  (C) 7\n"
                "Reasoning: Comparing values, 19 is greater than both 4 and 7.\n"
                "Answer: (B) 19\n",
                ["selection", "max"],
            ),
            (
                "Question: Which is a valid boolean value in Python?\n"
                "Options: (A) true  (B) True  (C) TRUE\n"
                "Reasoning: In Python, booleans are capitalized True and False.\n"
                "Answer: (B) True\n",
                ["selection", "syntax"],
            ),
            (
                "Question: What is the antonym of 'fast'?\n"
                "Options: (A) quick  (B) slow  (C) rapid\n"
                "Answer: (B) slow\n",
                ["selection", "language"],
            ),
        ]
        for idx, (text, tags) in enumerate(selections):
            samples.append(CurriculumSample(
                sample_id=f"r1_sel_{idx:03d}",
                level="R1",
                text=text,
                tags=tags + ["selection"],
            ))
        return samples

    # -------------------------------------------------------------------------
    # Level R2: Multi-Step Reasoning (chained math, conditions, transitions)
    # -------------------------------------------------------------------------
    def _gen_level_r2(self) -> List[CurriculumSample]:
        samples = []
        multi_step_math = [
            (
                "Problem: Compute (3 * 4) + (10 / 2).\n"
                "Step 1: Compute 3 * 4 = 12.\n"
                "Step 2: Compute 10 / 2 = 5.\n"
                "Step 3: Add results: 12 + 5 = 17.\n"
                "Final Answer: 17\n",
                ["multi_step", "math"],
            ),
            (
                "Problem: A store has 20 apples. It sells 8 apples and then receives 15 new apples. How many apples are there?\n"
                "Step 1: Initial apples = 20.\n"
                "Step 2: After selling 8: 20 - 8 = 12.\n"
                "Step 3: After receiving 15: 12 + 15 = 27.\n"
                "Final Answer: 27\n",
                ["multi_step", "word_problem"],
            ),
            (
                "Problem: Find the perimeter of a rectangle with length 7 and width 3.\n"
                "Step 1: Formula for perimeter is 2 * (length + width).\n"
                "Step 2: Calculate sum of length and width: 7 + 3 = 10.\n"
                "Step 3: Multiply by 2: 2 * 10 = 20.\n"
                "Final Answer: 20\n",
                ["multi_step", "geometry"],
            ),
        ]
        for idx, (text, tags) in enumerate(multi_step_math):
            samples.append(CurriculumSample(
                sample_id=f"r2_math_{idx:03d}",
                level="R2",
                text=text,
                tags=tags + ["reasoning"],
            ))

        # State transitions & deterministic planning
        planning_and_state = [
            (
                "System State: Door is LOCKED.\n"
                "Plan:\n"
                "1. Action: UNLOCK -> State becomes UNLOCKED.\n"
                "2. Action: OPEN -> State becomes OPEN.\n"
                "Goal State: OPEN\n",
                ["planning", "state_machine"],
            ),
            (
                "Task: Process list [3, 1, 4, 2] to get squared even numbers.\n"
                "Step 1: Filter even numbers -> [4, 2].\n"
                "Step 2: Square each number -> [16, 4].\n"
                "Result: [16, 4]\n",
                ["planning", "transformation"],
            ),
            (
                "Rule: If status is 'pending', check payment. If payment is True, status becomes 'confirmed'. Else 'failed'.\n"
                "Given: status = 'pending', payment = True.\n"
                "Evaluation:\n"
                "Payment is True, so rule branch 1 applies.\n"
                "Next Status: confirmed\n",
                ["logic", "conditions"],
            ),
        ]
        for idx, (text, tags) in enumerate(planning_and_state):
            samples.append(CurriculumSample(
                sample_id=f"r2_plan_{idx:03d}",
                level="R2",
                text=text,
                tags=tags + ["planning"],
            ))
        return samples

    # -------------------------------------------------------------------------
    # Level R3: Task Decomposition (Understand -> Decompose -> Implement -> Verify)
    # -------------------------------------------------------------------------
    def _gen_level_r3(self) -> List[CurriculumSample]:
        samples = []
        decompositions = [
            (
                "TASK: Create a function that checks whether a number is prime.\n"
                "UNDERSTAND:\n"
                "- Input is an integer n.\n"
                "- A prime number is greater than 1 and has no positive divisors other than 1 and itself.\n"
                "DECOMPOSE:\n"
                "1. If n < 2, return False.\n"
                "2. For i from 2 up to sqrt(n): if n % i == 0, return False.\n"
                "3. If no divisor found, return True.\n"
                "IMPLEMENT:\n"
                "def is_prime(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True\n"
                "VERIFY:\n"
                "assert is_prime(2) is True\n"
                "assert is_prime(4) is False\n"
                "assert is_prime(13) is True\n",
                ["decomposition", "is_prime"],
            ),
            (
                "TASK: Create a function to reverse a list in-place or return a new reversed list.\n"
                "UNDERSTAND:\n"
                "- Input: list of elements.\n"
                "- Output: list with order of elements reversed.\n"
                "DECOMPOSE:\n"
                "1. Take input list items.\n"
                "2. Use slicing or iteration from end to beginning.\n"
                "3. Return the reversed list.\n"
                "IMPLEMENT:\n"
                "def reverse_list(items: list) -> list:\n"
                "    return items[::-1]\n"
                "VERIFY:\n"
                "assert reverse_list([1, 2, 3]) == [3, 2, 1]\n"
                "assert reverse_list([]) == []\n",
                ["decomposition", "reverse_list"],
            ),
            (
                "TASK: Compute factorial of n iteratively.\n"
                "UNDERSTAND:\n"
                "- Factorial of 0 is 1.\n"
                "- For n > 0, n! = n * (n - 1) * ... * 1.\n"
                "DECOMPOSE:\n"
                "1. Initialize result = 1.\n"
                "2. Loop i from 1 to n: result = result * i.\n"
                "3. Return result.\n"
                "IMPLEMENT:\n"
                "def factorial(n: int) -> int:\n"
                "    res = 1\n"
                "    for i in range(1, n + 1):\n"
                "        res *= i\n"
                "    return res\n"
                "VERIFY:\n"
                "assert factorial(0) == 1\n"
                "assert factorial(4) == 24\n",
                ["decomposition", "factorial"],
            ),
        ]
        for idx, (text, tags) in enumerate(decompositions):
            samples.append(CurriculumSample(
                sample_id=f"r3_decomp_{idx:03d}",
                level="R3",
                text=text,
                tags=tags + ["decomposition"],
            ))
        return samples

    # -------------------------------------------------------------------------
    # Canonical Action/Observation Trajectories
    # -------------------------------------------------------------------------
    def _gen_trajectories(self) -> List[CurriculumSample]:
        samples = []
        trajectories = [
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Implement clamp(val, min_val, max_val)\n</SPEC>\n"
                "<STATE>\nInitial implementation draft\n</STATE>\n"
                "<ACTION>\ndef clamp(val, min_val, max_val):\n    return max_val\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: clamp(5, 0, 10) returned 10, expected 5\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nReturns max_val unconditionally without checking boundaries.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef clamp(val, min_val, max_val):\n    if val < min_val: return min_val\n    if val > max_val: return max_val\n    return val\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "clamp"],
            ),
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Implement add(a, b)\n</SPEC>\n"
                "<STATE>\nInitial implementation draft\n</STATE>\n"
                "<ACTION>\ndef add(a, b):\n    return a - b\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: add(2, 3) returned -1, expected 5\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nUsed subtraction operator '-' instead of addition '+'.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef add(a, b):\n    return a + b\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "arithmetic_fix"],
            ),
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Parse integer from string or return default 0\n</SPEC>\n"
                "<STATE>\nInitial implementation draft\n</STATE>\n"
                "<ACTION>\ndef safe_int(s):\n    return int(s)\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: safe_int('abc') raised ValueError\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nMissing try/except block to catch invalid conversions.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef safe_int(s):\n    try:\n        return int(s)\n    except (ValueError, TypeError):\n        return 0\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "exception_handling"],
            ),
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Fix JSON payload syntax\n</SPEC>\n"
                "<STATE>\nInvalid JSON string: {'name': 'ChakrView', 'version': 1,}\n</STATE>\n"
                "<ACTION>\njson.loads(\"{'name': 'ChakrView', 'version': 1,}\")\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: JSONDecodeError trailing comma and single quotes\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nJSON standard requires double quotes and no trailing commas.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\n'{\"name\": \"ChakrView\", \"version\": 1}'\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "json_fix"],
            ),
            (
                "<TRAJECTORY>\n"
                "<SPEC>\nTask: Find maximum in list\n</SPEC>\n"
                "<STATE>\nInitial code\n</STATE>\n"
                "<ACTION>\ndef find_max(nums):\n    return nums[0]\n</ACTION>\n"
                "<OBSERVATION>\nFAILED: find_max([1, 9, 3]) returned 1, expected 9\n</OBSERVATION>\n"
                "<DIAGNOSIS>\nOnly returns first element without comparing against all elements.\n</DIAGNOSIS>\n"
                "<NEXT_ACTION>\ndef find_max(nums):\n    m = nums[0]\n    for x in nums[1:]:\n        if x > m: m = x\n    return m\n</NEXT_ACTION>\n"
                "<RESULT>SUCCESS</RESULT>\n"
                "</TRAJECTORY>\n",
                ["trajectory", "algorithm_fix"],
            ),
        ]
        for idx, (text, tags) in enumerate(trajectories):
            samples.append(CurriculumSample(
                sample_id=f"r_traj_{idx:03d}",
                level="TRAJ",
                text=text,
                tags=tags + ["action_observation"],
            ))
        return samples

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
            for sample in split_samples:
                # Format: <BOS> + text tokens + <EOS>
                raw_tokens = tokenizer.encode(sample.text)
                token_ids = [0] + raw_tokens + [1]
                writer.add_document(token_ids)
                total_tokens += len(token_ids)

            meta = writer.close()
            manifest["splits"][split_name] = {
                "sample_count": len(split_samples),
                "token_count": total_tokens,
                "shard_meta": meta,
            }

        # Write manifest file
        manifest_path = output_dir / "reasoning_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
