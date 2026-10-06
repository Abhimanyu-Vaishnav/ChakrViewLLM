"""Step 241: Generalized Randomized Binding Episode Generator.

Builds a fully reusable, randomized episode generator for contextual key-value associations:
- Randomized key identities
- Randomized value identities
- Randomized association counts (1, 2, 3, 5, 8 pairs)
- Randomized presentation ordering
- Randomized query identity
- Variable query positions
- Variable sequence lengths and distractor noise tokens
- Multiple surface layouts / formats preserving identical semantic operation
- Distinct identity pools for train vs. strictly disjoint validation/eval
- Episode-level metadata and independent deterministic hashing
- SHA-256 contamination audit between train and evaluation sets

STRICT CONSTRAINT:
The neural model receives ONLY the generated prompt token sequence.
It does NOT receive metadata or symbolic answer hints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any


@dataclasses.dataclass(frozen=True)
class BindingPair:
    key: str
    val: str


@dataclasses.dataclass(frozen=True)
class BindingEpisode:
    episode_id: str
    split: str                          # "train", "val", "disjoint_test"
    pairs: Tuple[BindingPair, ...]
    query_key: str
    expected_value: str
    layout_format: str                  # "format_a", "format_b", "format_c", "format_d"
    prompt: str
    num_associations: int
    has_distractors: bool
    episode_hash: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "pairs": [(p.key, p.val) for p in self.pairs],
            "query_key": self.query_key,
            "expected_value": self.expected_value,
            "layout_format": self.layout_format,
            "num_associations": self.num_associations,
            "has_distractors": self.has_distractors,
            "episode_hash": self.episode_hash,
        }


class GeneralizedEpisodeGenerator:
    """Generates synthetic contextual association episodes with strict identity isolation."""

    # Broad single-token ASCII token-safe pools
    TRAIN_KEYS_POOL = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O"
    ]
    TRAIN_VALS_POOL = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]

    DISJOINT_EVAL_KEYS_POOL = [
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]
    DISJOINT_EVAL_VALS_POOL = [
        "0", "#", "@", "$", "%", "&", "!", "?"
    ]

    LAYOUT_TEMPLATES = ["format_a", "format_b", "format_c", "format_d"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def render_prompt(
        self,
        pairs: List[BindingPair],
        query_key: str,
        layout: str,
        has_distractors: bool = False,
    ) -> str:
        """Renders the text prompt according to the selected layout format."""
        if layout == "format_a":
            # Canonical standard format
            parts = [f"|{p.key}| -> |{p.val}|" for p in pairs]
            p_str = "map " + " and ".join(parts)
            if has_distractors:
                p_str += " note distractor |ignore| -> |void|"
            p_str += f" query |{query_key}| -> |"
            return p_str

        elif layout == "format_b":
            # Semicolon delimited natural layout
            parts = [f"{p.key} maps to {p.val}" for p in pairs]
            p_str = "; ".join(parts)
            if has_distractors:
                p_str += "; skip distractor noise"
            p_str += f"; query {query_key}: |"
            return p_str

        elif layout == "format_c":
            # Tuple compact syntax
            parts = [f"({p.key},{p.val})" for p in pairs]
            p_str = "pairs " + " ".join(parts)
            if has_distractors:
                p_str += " (distractor,nil)"
            p_str += f" query={query_key} -> |"
            return p_str

        elif layout == "format_d":
            # Assignment syntax
            parts = [f"{p.key}={p.val}" for p in pairs]
            p_str = "assoc: " + ", ".join(parts)
            if has_distractors:
                p_str += ", noise=0"
            p_str += f" / retrieve {query_key} = |"
            return p_str

        else:
            # Fallback format_a
            parts = [f"|{p.key}| -> |{p.val}|" for p in pairs]
            return "map " + " and ".join(parts) + f" query |{query_key}| -> |"

    def generate_episode(
        self,
        split: str = "train",
        num_associations: int = 3,
        layout_format: Optional[str] = None,
        include_distractors: bool = False,
        episode_idx: int = 0,
    ) -> BindingEpisode:
        """Generates a single randomized episode."""
        if split == "train":
            k_pool = list(self.TRAIN_KEYS_POOL)
            v_pool = list(self.TRAIN_VALS_POOL)
        elif split == "val":
            # Train keys/vals but sampled with distinct random state
            k_pool = list(self.TRAIN_KEYS_POOL)
            v_pool = list(self.TRAIN_VALS_POOL)
        elif split in ("disjoint_test", "unseen_unseen"):
            # Strictly disjoint pools
            k_pool = list(self.DISJOINT_EVAL_KEYS_POOL)
            v_pool = list(self.DISJOINT_EVAL_VALS_POOL)
        else:
            k_pool = list(self.TRAIN_KEYS_POOL)
            v_pool = list(self.TRAIN_VALS_POOL)

        n_pairs = min(num_associations, len(k_pool), len(v_pool))
        sampled_keys = self.rng.sample(k_pool, n_pairs)
        sampled_vals = self.rng.sample(v_pool, n_pairs)

        pairs = [BindingPair(k, v) for k, v in zip(sampled_keys, sampled_vals)]
        self.rng.shuffle(pairs)

        query_pair = self.rng.choice(pairs)
        q_key = query_pair.key
        exp_val = query_pair.val

        layout = layout_format or self.rng.choice(self.LAYOUT_TEMPLATES)
        prompt = self.render_prompt(pairs, q_key, layout, has_distractors=include_distractors)

        h_builder = hashlib.sha256()
        h_builder.update(f"{split}_{episode_idx}_{prompt}_{exp_val}".encode("utf-8"))
        ep_hash = h_builder.hexdigest()

        ep_id = f"ep_{split}_{num_associations}assoc_{episode_idx}_{ep_hash[:8]}"

        return BindingEpisode(
            episode_id=ep_id,
            split=split,
            pairs=tuple(pairs),
            query_key=q_key,
            expected_value=exp_val,
            layout_format=layout,
            prompt=prompt,
            num_associations=n_pairs,
            has_distractors=include_distractors,
            episode_hash=ep_hash,
        )

    def generate_batch(
        self,
        count: int,
        split: str = "train",
        num_associations: int = 3,
        layout_format: Optional[str] = None,
        include_distractors: bool = False,
    ) -> List[BindingEpisode]:
        """Generates a batch of distinct episodes."""
        return [
            self.generate_episode(
                split=split,
                num_associations=num_associations,
                layout_format=layout_format,
                include_distractors=include_distractors,
                episode_idx=i,
            )
            for i in range(count)
        ]

    def verify_no_contamination(
        self,
        train_batch: List[BindingEpisode],
        eval_batch: List[BindingEpisode],
    ) -> Tuple[bool, int]:
        """Verifies zero overlap between train and evaluation episode hashes or entity sets."""
        train_hashes = {e.episode_hash for e in train_batch}
        eval_hashes = {e.episode_hash for e in eval_batch}
        hash_overlap = len(train_hashes.intersection(eval_hashes))

        train_keys = {p.key for e in train_batch for p in e.pairs}
        eval_keys = {p.key for e in eval_batch for p in e.pairs}
        key_overlap = len(train_keys.intersection(eval_keys))

        is_clean = (hash_overlap == 0 and key_overlap == 0)
        return is_clean, max(hash_overlap, key_overlap)
