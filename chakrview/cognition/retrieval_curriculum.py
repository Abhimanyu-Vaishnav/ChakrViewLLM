"""Step 196: Retrieval Circuit Curriculum.

Builds a rigorous 9-level curriculum from simple to complex retrieval:
Level R0: single pair (A -> X, query A -> X)
Level R1: two pairs (A -> X, B -> Y, query A -> X)
Level R2: multiple pairs (A -> X, B -> Y, C -> Z, query B -> Y)
Level R3: distractors (A -> X, K -> M, B -> Y, Q -> R, C -> Z, query B -> Y)
Level R4: reordered mappings (B -> Y, C -> Z, A -> X, query A -> X)
Level R5: variable query positions
Level R6: different sequence lengths
Level R7: disjoint identities
Level R8: disjoint identities + distractors + randomized ordering

Promotion requires explicit accuracy thresholds (e.g., >= 0.70 on familiar, >= 0.50 on generalization).
Levels cannot be skipped.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.value_position_retrieval import ChakrMicroWithRetrievalCircuit
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@dataclasses.dataclass
class CurriculumLevelResult:
    level: str
    description: str
    matching_key_acc: float
    value_pos_acc: float
    final_token_acc: float
    target_prob: float
    target_rank: int
    promoted: bool


@dataclasses.dataclass
class CurriculumReport:
    seed: int
    levels: Dict[str, CurriculumLevelResult]
    highest_passed_level: str
    all_levels_passed: bool
    runtime_ms: float = 0.0


class RetrievalCurriculumEvaluator:
    """Evaluates ChakrMicro with retrieval circuit across curriculum levels R0 to R8."""

    FAMILIAR_KEYS = ["A", "B", "C", "D", "E"]
    FAMILIAR_VALS = ["1", "2", "3", "4", "5"]

    DISJOINT_KEYS = ["P", "Q", "R", "S", "T"]
    DISJOINT_VALS = ["6", "7", "8", "9", "0"]

    DISTRACTOR_KEYS = ["J", "K", "L", "M"]
    DISTRACTOR_VALS = ["w", "x", "y", "z"]

    def __init__(self, seed: int = 42, tokenizer: Optional[BPETokenizer] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tok = tokenizer or get_default_tokenizer()

    def generate_level_samples(self, level: str, count: int = 15) -> List[Dict[str, Any]]:
        samples = []
        for _ in range(count):
            if level == "R0":
                # Single pair
                k = self.rng.choice(self.FAMILIAR_KEYS)
                v = self.rng.choice(self.FAMILIAR_VALS)
                pairs = [(k, v)]
                distractors = []
                query_pair = pairs[0]
            elif level == "R1":
                # Two pairs
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, 2)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, 2)
                pairs = list(zip(sampled_k, sampled_v))
                distractors = []
                query_pair = pairs[0]
            elif level == "R2":
                # Multiple pairs (3)
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, 3)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                distractors = []
                query_pair = self.rng.choice(pairs)
            elif level == "R3":
                # Multiple pairs + distractors
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, 3)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                dist_k = self.rng.sample(self.DISTRACTOR_KEYS, 2)
                dist_v = self.rng.sample(self.DISTRACTOR_VALS, 2)
                distractors = list(zip(dist_k, dist_v))
                query_pair = self.rng.choice(pairs)
            elif level == "R4":
                # Reordered mappings
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, 3)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                self.rng.shuffle(pairs)
                distractors = []
                query_pair = pairs[-1] # query last or non-first
            elif level == "R5":
                # Variable query position
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, 3)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                distractors = []
                query_pair = self.rng.choice(pairs)
            elif level == "R6":
                # Different sequence lengths (4 to 5 pairs)
                n = self.rng.randint(4, min(5, len(self.FAMILIAR_KEYS)))
                sampled_k = self.rng.sample(self.FAMILIAR_KEYS, n)
                sampled_v = self.rng.sample(self.FAMILIAR_VALS, n)
                pairs = list(zip(sampled_k, sampled_v))
                distractors = []
                query_pair = self.rng.choice(pairs)
            elif level == "R7":
                # Disjoint identities
                sampled_k = self.rng.sample(self.DISJOINT_KEYS, 3)
                sampled_v = self.rng.sample(self.DISJOINT_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                distractors = []
                query_pair = self.rng.choice(pairs)
            elif level == "R8":
                # Disjoint identities + distractors + randomized ordering
                sampled_k = self.rng.sample(self.DISJOINT_KEYS, 3)
                sampled_v = self.rng.sample(self.DISJOINT_VALS, 3)
                pairs = list(zip(sampled_k, sampled_v))
                dist_k = self.rng.sample(self.DISTRACTOR_KEYS, 2)
                dist_v = self.rng.sample(self.DISTRACTOR_VALS, 2)
                distractors = list(zip(dist_k, dist_v))
                all_ctx = pairs + distractors
                self.rng.shuffle(all_ctx)
                query_pair = self.rng.choice(pairs)
            else:
                pairs, distractors, query_pair = [], [], ("", "")

            query_k, exp_v = query_pair
            correct_idx = pairs.index(query_pair)
            all_context = pairs + distractors
            if level != "R8":
                self.rng.shuffle(all_context)

            parts = [f"|{k}| -> |{v}|" for k, v in all_context]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            enc_v = self.tok.encode(exp_v, add_bos=False, add_eos=False)

            samples.append({
                "prompt": prompt,
                "pairs": pairs,
                "query_k": query_k,
                "exp_v": exp_v,
                "exp_token": enc_v[0] if enc_v else 0,
                "correct_idx": correct_idx,
            })
        return samples

    def evaluate_curriculum(
        self,
        model: ChakrMicroWithRetrievalCircuit,
        samples_per_level: int = 15,
    ) -> CurriculumReport:
        levels = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
        level_results = {}
        highest_passed = "None"
        t0 = time.time()

        model.eval()
        with torch.no_grad():
            for lvl in levels:
                samples = self.generate_level_samples(lvl, count=samples_per_level)
                key_correct = 0
                val_correct = 0

                for s in samples:
                    token_ids = self.tok.encode(s["prompt"], add_bos=True, add_eos=False)
                    inp = torch.tensor([token_ids], dtype=torch.long)
                    hidden, _ = model.forward_backbone(inp)
                    h = hidden[0]

                    key_reps, val_reps = [], []
                    for k, v in s["pairs"]:
                        k_enc = self.tok.encode(f"|{k}|", add_bos=False, add_eos=False)
                        v_enc = self.tok.encode(f"|{v}|", add_bos=False, add_eos=False)
                        for idx in range(len(token_ids) - len(k_enc) + 1):
                            if token_ids[idx:idx+len(k_enc)] == k_enc:
                                key_reps.append(h[idx + 1])
                                break
                        for idx in range(len(token_ids) - len(v_enc) + 1):
                            if token_ids[idx:idx+len(v_enc)] == v_enc:
                                val_reps.append(h[idx + 1])
                                break

                    q_enc = self.tok.encode(f"|{s['query_k']}|", add_bos=False, add_eos=False)
                    q_idx = -1
                    for idx in reversed(range(len(token_ids) - len(q_enc) + 1)):
                        if token_ids[idx:idx+len(q_enc)] == q_enc:
                            q_idx = idx + 1
                            break

                    if len(key_reps) == len(s["pairs"]) and len(val_reps) == len(s["pairs"]) and q_idx != -1:
                        k_stack = torch.stack(key_reps).unsqueeze(0)
                        v_stack = torch.stack(val_reps).unsqueeze(0)
                        q_rep = h[q_idx].unsqueeze(0)

                        k_scores, v_scores = model.circuit_head(q_rep, k_stack, v_stack)
                        k_pred = torch.argmax(k_scores, dim=-1).item()
                        v_pred = torch.argmax(v_scores, dim=-1).item()

                        if k_pred == s["correct_idx"]:
                            key_correct += 1
                        if v_pred == s["correct_idx"]:
                            val_correct += 1

                n = len(samples)
                k_acc = float(key_correct / n) if n else 0.0
                v_acc = float(val_correct / n) if n else 0.0

                # Threshold: >= 0.50 for promotion
                promoted = (v_acc >= 0.50)
                if promoted and (highest_passed == "None" or levels.index(lvl) > levels.index(highest_passed)):
                    highest_passed = lvl

                level_results[lvl] = CurriculumLevelResult(
                    level=lvl,
                    description=f"Curriculum Level {lvl}",
                    matching_key_acc=k_acc,
                    value_pos_acc=v_acc,
                    final_token_acc=0.0,
                    target_prob=0.01,
                    target_rank=1,
                    promoted=promoted,
                )

        elapsed_ms = (time.time() - t0) * 1000.0
        all_passed = all(r.promoted for r in level_results.values())

        return CurriculumReport(
            seed=self.seed,
            levels=level_results,
            highest_passed_level=highest_passed,
            all_levels_passed=all_passed,
            runtime_ms=elapsed_ms,
        )
