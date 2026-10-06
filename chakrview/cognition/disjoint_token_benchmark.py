"""Step 186: Disjoint Token Retrieval Control Benchmark.

Provides a rigorous, randomized benchmark fixture evaluating:
1. Known key + known value (in-distribution associative retrieval)
2. Known key + unseen value
3. Unseen key + known value
4. Unseen key + unseen value (fully disjoint)

Includes permutation of mapping order, query positions, distractors,
variable lengths, and strict train/test contamination hashing.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclasses.dataclass
class DisjointBenchmarkSample:
    prompt: str
    expected_token: int
    expected_char: str
    category: str  # "known_known", "known_unseen", "unseen_known", "unseen_unseen"
    mapping_pairs: List[Tuple[str, str]]
    query_key: str
    num_distractors: int
    distractors: List[Tuple[str, str]]


@dataclasses.dataclass
class DisjointSplitResults:
    accuracy: float
    mean_target_prob: float
    mean_target_logit: float
    median_rank: float
    sample_count: int


@dataclasses.dataclass
class DisjointBenchmarkReport:
    seed: int
    contamination_hash: str
    known_known: DisjointSplitResults
    known_unseen: DisjointSplitResults
    unseen_known: DisjointSplitResults
    unseen_unseen: DisjointSplitResults
    overall_accuracy: float


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


class DisjointRetrievalFixture:
    """Rigorous generator for disjoint associative retrieval tasks."""

    KNOWN_KEYS = ["A", "B", "C", "D", "E"]
    KNOWN_VALUES = ["1", "2", "3", "4", "5"]

    UNSEEN_KEYS = ["P", "Q", "R", "S", "T"]
    UNSEEN_VALUES = ["7", "8", "9", "0", "6"]

    DISTRACTOR_KEYS = ["J", "K", "L", "M"]
    DISTRACTOR_VALUES = ["w", "x", "y", "z"]

    def __init__(self, seed: int = 42, tokenizer: Optional[BPETokenizer] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer or get_default_tokenizer()

    def generate_benchmark_suite(
        self,
        samples_per_split: int = 20,
    ) -> Dict[str, List[DisjointBenchmarkSample]]:
        """Generate balanced suites for all 4 disjoint splits."""
        splits = {
            "known_known": [],
            "known_unseen": [],
            "unseen_known": [],
            "unseen_unseen": [],
        }

        for category, (keys_pool, vals_pool) in [
            ("known_known", (self.KNOWN_KEYS, self.KNOWN_VALUES)),
            ("known_unseen", (self.KNOWN_KEYS, self.UNSEEN_VALUES)),
            ("unseen_known", (self.UNSEEN_KEYS, self.KNOWN_VALUES)),
            ("unseen_unseen", (self.UNSEEN_KEYS, self.UNSEEN_VALUES)),
        ]:
            for _ in range(samples_per_split):
                num_pairs = self.rng.randint(2, min(4, len(keys_pool)))
                sampled_keys = self.rng.sample(keys_pool, num_pairs)
                sampled_vals = self.rng.sample(vals_pool, num_pairs)
                pairs = list(zip(sampled_keys, sampled_vals))
                self.rng.shuffle(pairs)

                num_dist = self.rng.randint(0, 2)
                dist_keys = self.rng.sample(self.DISTRACTOR_KEYS, num_dist)
                dist_vals = self.rng.sample(self.DISTRACTOR_VALUES, num_dist)
                distractors = list(zip(dist_keys, dist_vals))

                all_context = pairs + distractors
                self.rng.shuffle(all_context)

                query_pair = self.rng.choice(pairs)
                query_k, query_v = query_pair

                parts = [f"|{k}| -> |{v}|" for k, v in all_context]
                prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"

                encoded_v = self.tokenizer.encode(query_v, add_bos=False, add_eos=False)
                exp_token = encoded_v[0]

                sample = DisjointBenchmarkSample(
                    prompt=prompt,
                    expected_token=exp_token,
                    expected_char=query_v,
                    category=category,
                    mapping_pairs=pairs,
                    query_key=query_k,
                    num_distractors=num_dist,
                    distractors=distractors,
                )
                splits[category].append(sample)

        return splits

    def evaluate_model(
        self,
        model: Any,
        samples_per_split: int = 15,
    ) -> DisjointBenchmarkReport:
        """Evaluate a model on the 4-split benchmark."""
        suite = self.generate_benchmark_suite(samples_per_split=samples_per_split)

        hasher = hashlib.sha256()
        for cat in sorted(suite.keys()):
            for s in suite[cat]:
                hasher.update(s.prompt.encode("utf-8"))
                hasher.update(str(s.expected_token).encode("utf-8"))
        contamination_hash = hasher.hexdigest()

        results_by_cat = {}
        total_correct = 0
        total_count = 0

        model.eval()
        with torch.no_grad():
            for cat, samples in suite.items():
                correct = 0
                probs = []
                logits_list = []
                ranks = []

                for s in samples:
                    tokens = self.tokenizer.encode(s.prompt, add_bos=True, add_eos=False)
                    input_ids = torch.tensor([tokens], dtype=torch.long)
                    out = model(input_ids)
                    if isinstance(out, tuple):
                        logits = out[0]
                    else:
                        logits = out

                    last_logits = logits[0, -1, :]
                    p = F.softmax(last_logits, dim=-1)

                    t_logit = float(last_logits[s.expected_token].item())
                    t_prob = float(p[s.expected_token].item())

                    sorted_idx = torch.argsort(last_logits, descending=True)
                    rank = int((sorted_idx == s.expected_token).nonzero(as_tuple=True)[0].item()) + 1

                    logits_list.append(t_logit)
                    probs.append(t_prob)
                    ranks.append(rank)

                    pred = int(torch.argmax(last_logits).item())
                    if pred == s.expected_token:
                        correct += 1

                n = len(samples)
                total_correct += correct
                total_count += n
                results_by_cat[cat] = DisjointSplitResults(
                    accuracy=float(correct / n) if n else 0.0,
                    mean_target_prob=float(sum(probs) / n) if n else 0.0,
                    mean_target_logit=float(sum(logits_list) / n) if n else 0.0,
                    median_rank=float(sorted(ranks)[len(ranks) // 2]) if n else 0.0,
                    sample_count=n,
                )

        return DisjointBenchmarkReport(
            seed=self.seed,
            contamination_hash=contamination_hash,
            known_known=results_by_cat["known_known"],
            known_unseen=results_by_cat["known_unseen"],
            unseen_known=results_by_cat["unseen_known"],
            unseen_unseen=results_by_cat["unseen_unseen"],
            overall_accuracy=float(total_correct / total_count) if total_count else 0.0,
        )
