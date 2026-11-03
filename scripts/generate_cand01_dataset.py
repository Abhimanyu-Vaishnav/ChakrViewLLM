"""
CAND_01 Synthetic Dataset Generation Infrastructure.
Specification: docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.3_REVIEWED.md
Authoritative Errata: docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md

GOVERNANCE RESTRICTIONS:
- Strictly operates on SYNTHETIC vocabularies and entity pools by default.
- CANNOT read canonical pretraining shards (Stage C) or vocabularies.
- Output generation is SAFE and MOCKED by default; production dataset creation is
  strictly unauthorized and disabled.
- Enforces zero partition contamination, bidirectional association exclusion (VER-03),
  exact prompt hash uniqueness, and deterministic record generation across all 8 Task Families (TF-01 to TF-08).
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np


class TaskFamily(str, Enum):
    TF01_DIRECT_COPY = "TF_01_DIRECT_COPY"
    TF02_SINGLE_PAIR = "TF_02_SINGLE_PAIR"
    TF03_MULTI_PAIR = "TF_03_MULTI_PAIR"
    TF04_DISTRACTOR = "TF_04_DISTRACTOR"
    TF05_SIMILAR_KEY = "TF_05_SIMILAR_KEY"
    TF06_RECOMBINATION = "TF_06_RECOMBINATION"
    TF07_OFFSET_SCALING = "TF_07_OFFSET_SCALING"
    TF08_CONTEXT_SCALING = "TF_08_CONTEXT_SCALING"


class PartitionType(str, Enum):
    PARTITION_A_TRAIN = "train"
    PARTITION_B_VAL_RECOMB = "val_recomb"
    PARTITION_C_TEST_RECOMB = "test_recomb"
    PARTITION_D_TEST_DISJOINT = "test_disjoint"


@dataclass
class DatasetRecord:
    """Canonical schema for a single CAND_01 experiment record."""
    record_id: str
    task_family: str
    partition: str
    prompt_text: str
    target_token: str
    target_token_id: int
    context_pairs: List[Tuple[str, str]]
    context_pair_ids: List[Tuple[int, int]]
    query_key: str
    query_key_id: int
    distractor_count: int
    key_val_distance: int
    prompt_length_tokens: int
    prompt_sha256: str = field(init=False)

    def __post_init__(self):
        self.prompt_sha256 = hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_family": self.task_family,
            "partition": self.partition,
            "prompt_text": self.prompt_text,
            "target_token": self.target_token,
            "target_token_id": self.target_token_id,
            "context_pairs": self.context_pairs,
            "context_pair_ids": self.context_pair_ids,
            "query_key": self.query_key,
            "query_key_id": self.query_key_id,
            "distractor_count": self.distractor_count,
            "key_val_distance": self.key_val_distance,
            "prompt_length_tokens": self.prompt_length_tokens,
            "prompt_sha256": self.prompt_sha256,
        }


class SyntheticEntityPool:
    """
    Constructs isolated synthetic entity token pools for training and disjoint testing.
    Default synthetic sizes: |E_train| = 450, |E_test| = 150 (Total: 600 synthetic tokens).
    Token IDs are assigned deterministically in a synthetic ID space [1000..1599].
    """
    def __init__(
        self,
        n_train: int = 450,
        n_test: int = 150,
        seed: int = 42,
        base_id: int = 1000,
    ):
        self.n_train = n_train
        self.n_test = n_test
        self.base_id = base_id
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Generate synthetic entity tokens e.g. "ent_1000", "ent_1001", ...
        self.train_tokens: List[Tuple[int, str]] = [
            (base_id + i, f"alpha_{base_id + i}") for i in range(n_train)
        ]
        self.test_tokens: List[Tuple[int, str]] = [
            (base_id + n_train + i, f"omega_{base_id + n_train + i}") for i in range(n_test)
        ]

        self.train_token_ids = set(t[0] for t in self.train_tokens)
        self.test_token_ids = set(t[0] for t in self.test_tokens)

        # Verification of token pool disjointness invariant
        assert len(self.train_token_ids.intersection(self.test_token_ids)) == 0, (
            "FATAL: E_train and E_test must be strictly disjoint!"
        )

        # Assign synthetic frequency stratification tiers
        all_ids = list(self.train_token_ids)
        rng.shuffle(all_ids)
        t_len = len(all_ids) // 3
        self.freq_tier1 = set(all_ids[:t_len])
        self.freq_tier2 = set(all_ids[t_len:2*t_len])
        self.freq_tier3 = set(all_ids[2*t_len:])


class CAND01DatasetGenerator:
    """
    Generates synthetic benchmark datasets for CAND_01 with strict cryptographic
    leakage checking, bidirectional exclusion, and schema validation across TF01-TF08.
    """
    def __init__(
        self,
        entity_pool: SyntheticEntityPool,
        seed: int = 999042,
    ):
        self.pool = entity_pool
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Storage for association closures: (k, v) directed pairs
        self.seen_pairs: Dict[str, Set[Tuple[int, int]]] = {
            PartitionType.PARTITION_A_TRAIN.value: set(),
            PartitionType.PARTITION_B_VAL_RECOMB.value: set(),
            PartitionType.PARTITION_C_TEST_RECOMB.value: set(),
            PartitionType.PARTITION_D_TEST_DISJOINT.value: set(),
        }
        # Symmetric closure {(k, v), (v, k)} for training (VER-03)
        self.train_symmetric_closure: Set[Tuple[int, int]] = set()

        # Prompt hashes across all partitions
        self.prompt_hashes: Dict[str, Set[str]] = {
            p.value: set() for p in PartitionType
        }

    def _sample_pairs(
        self,
        token_pool: List[Tuple[int, str]],
        num_pairs: int,
        partition: PartitionType,
    ) -> List[Tuple[Tuple[int, str], Tuple[int, str]]]:
        """
        Samples unique pairs ensuring:
        1. k != v
        2. No duplicate keys within the sampled set (unique key constraint for unambiguous retrieval).
        3. No overlap with previously generated directed pairs in this partition.
        4. Enforces Bidirectional Association Exclusion (VER-03):
           If partition is Val Recomb or Test Recomb, neither (k, v) nor (v, k)
           may have been present in Train!
        """
        pairs: List[Tuple[Tuple[int, str], Tuple[int, str]]] = []
        n_tokens = len(token_pool)
        attempts = 0
        max_attempts = num_pairs * 1000
        used_keys_in_call: Set[int] = set()

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            idx_k, idx_v = self.rng.choice(n_tokens, size=2, replace=False)
            k_tuple = token_pool[idx_k]
            v_tuple = token_pool[idx_v]
            k_id, v_id = k_tuple[0], v_tuple[0]

            if k_id in used_keys_in_call:
                continue

            pair_tuple = (k_id, v_id)

            # In-partition uniqueness
            if pair_tuple in self.seen_pairs[partition.value]:
                continue

            # Bidirectional exclusion against Train symmetric closure (VER-03)
            if partition in [PartitionType.PARTITION_B_VAL_RECOMB, PartitionType.PARTITION_C_TEST_RECOMB]:
                if pair_tuple in self.train_symmetric_closure:
                    continue
                if (v_id, k_id) in self.train_symmetric_closure:
                    continue

            # Register pair
            self.seen_pairs[partition.value].add(pair_tuple)
            used_keys_in_call.add(k_id)

            if partition == PartitionType.PARTITION_A_TRAIN:
                self.train_symmetric_closure.add((k_id, v_id))
                self.train_symmetric_closure.add((v_id, k_id))

            pairs.append((k_tuple, v_tuple))

        if len(pairs) < num_pairs:
            raise RuntimeError(
                f"Failed to sample {num_pairs} pairs for {partition.value} after {attempts} attempts. "
                "Combinatorial capacity exhausted."
            )
        return pairs

    def generate_record(
        self,
        task_family: TaskFamily,
        partition: PartitionType,
        index: int,
    ) -> DatasetRecord:
        """Constructs a single synthetic record adhering to prompt template formatting for all TF-01 to TF-08."""
        # Enforce TF-01 exclusion rule from Partitions B & C per v1.3 / v1.3.1
        if task_family == TaskFamily.TF01_DIRECT_COPY and partition in [
            PartitionType.PARTITION_B_VAL_RECOMB,
            PartitionType.PARTITION_C_TEST_RECOMB,
        ]:
            raise ValueError(
                f"TF-01 Direct Copy is strictly EXCLUDED from {partition.value} per v1.3 / v1.3.1 specification!"
            )

        # Select appropriate token pool
        if partition == PartitionType.PARTITION_D_TEST_DISJOINT:
            pool = self.pool.test_tokens
        else:
            pool = self.pool.train_tokens

        # ----------------------------------------------------------------------
        # TF-01: Direct Identity Copy: COPY: X -> X
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF01_DIRECT_COPY:
            token_idx = index % len(pool)
            tok_id, tok_str = pool[token_idx]
            prompt = f"COPY: {tok_str} ->"
            record = DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tok_str,
                target_token_id=tok_id,
                context_pairs=[(tok_str, tok_str)],
                context_pair_ids=[(tok_id, tok_id)],
                query_key=tok_str,
                query_key_id=tok_id,
                distractor_count=0,
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )
            return record

        # ----------------------------------------------------------------------
        # TF-02: Single Key-Value Association: PAIR: K=V. QUERY: K -> V
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF02_SINGLE_PAIR:
            sampled = self._sample_pairs(pool, 1, partition)
            (k_id, k_str), (v_id, v_str) = sampled[0]
            prompt = f"PAIR: {k_str}={v_str}. QUERY: {k_str} ->"
            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=v_str,
                target_token_id=v_id,
                context_pairs=[(k_str, v_str)],
                context_pair_ids=[(k_id, v_id)],
                query_key=k_str,
                query_key_id=k_id,
                distractor_count=0,
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-03: Multi-Pair Association: PAIRS: K1=V1, K2=V2... QUERY: K_tgt -> V_tgt
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF03_MULTI_PAIR:
            n_pairs = 3  # 1 target + 2 distractors
            sampled = self._sample_pairs(pool, n_pairs, partition)
            target_pair = sampled[0]
            distractors = sampled[1:]
            all_pairs = [target_pair] + distractors
            self.rng.shuffle(all_pairs)

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            body = ", ".join([f"{k}={v}" for (k_id, k), (v_id, v) in all_pairs])
            prompt = f"PAIRS: {body}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(k, v) for (k_id, k), (v_id, v) in all_pairs],
                context_pair_ids=[(k_id, v_id) for (k_id, k), (v_id, v) in all_pairs],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=len(distractors),
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-04: Distractor Resistance: Context with 1, 3, 7 distractors
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF04_DISTRACTOR:
            # Vary distractor levels deterministically
            d_levels = [1, 3, 7]
            n_dist = d_levels[index % len(d_levels)]
            sampled = self._sample_pairs(pool, 1 + n_dist, partition)
            target_pair = sampled[0]
            distractors = sampled[1:]
            all_pairs = [target_pair] + distractors
            self.rng.shuffle(all_pairs)

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            body = ", ".join([f"{k}={v}" for (k_id, k), (v_id, v) in all_pairs])
            prompt = f"PAIRS: {body}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(k, v) for (k_id, k), (v_id, v) in all_pairs],
                context_pair_ids=[(k_id, v_id) for (k_id, k), (v_id, v) in all_pairs],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=n_dist,
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-05: Similar-Key Discrimination: Shared Prefix Collisions
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF05_SIMILAR_KEY:
            sampled = self._sample_pairs(pool, 3, partition)
            target_pair = sampled[0]
            distractor1 = sampled[1]
            distractor2 = sampled[2]

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            # Form prefix collision by synthetic naming tag
            similar_q_str = f"{q_str}_alt"
            (d1_id, d1_k), (d1_vid, d1_v) = distractor1
            (d2_id, d2_k), (d2_vid, d2_v) = distractor2

            # All pairs: target, near-key distractor, other distractor
            all_pairs_display = [(q_str, tgt_str), (similar_q_str, d1_v), (d2_k, d2_v)]
            self.rng.shuffle(all_pairs_display)

            body = ", ".join([f"{k}={v}" for k, v in all_pairs_display])
            prompt = f"PAIRS: {body}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(q_str, tgt_str), (d1_k, d1_v), (d2_k, d2_v)],
                context_pair_ids=[(q_id, tgt_id), (d1_id, d1_vid), (d2_id, d2_vid)],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=2,
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-06: Unseen Pair Recombination
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF06_RECOMBINATION:
            sampled = self._sample_pairs(pool, 3, partition)
            target_pair = sampled[0]
            distractors = sampled[1:]
            all_pairs = [target_pair] + distractors
            self.rng.shuffle(all_pairs)

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            body = ", ".join([f"{k}={v}" for (k_id, k), (v_id, v) in all_pairs])
            prompt = f"RECOMB_PAIRS: {body}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(k, v) for (k_id, k), (v_id, v) in all_pairs],
                context_pair_ids=[(k_id, v_id) for (k_id, k), (v_id, v) in all_pairs],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=2,
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-07: Context-Position / Offset Scaling: Distance padded between target and query
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF07_OFFSET_SCALING:
            offset_bands = [10, 25, 50, 100]
            pad_tokens_count = offset_bands[index % len(offset_bands)]

            sampled = self._sample_pairs(pool, 2, partition)
            target_pair = sampled[0]
            distractor = sampled[1]

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            (d_id, d_k), (d_vid, d_v) = distractor

            # Construct distance padding tokens
            pad_fill = " ".join([f"pad_{p % 10}" for p in range(pad_tokens_count)])
            prompt = f"PAIR: {q_str}={tgt_str}. CONTEXT_BUFFER: {pad_fill}. DISTRACTOR: {d_k}={d_v}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(q_str, tgt_str), (d_k, d_v)],
                context_pair_ids=[(q_id, tgt_id), (d_id, d_vid)],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=1,
                key_val_distance=pad_tokens_count + 5,
                prompt_length_tokens=len(prompt.split()),
            )

        # ----------------------------------------------------------------------
        # TF-08: Context-Length Scaling: Dense context with N pairs
        # ----------------------------------------------------------------------
        if task_family == TaskFamily.TF08_CONTEXT_SCALING:
            dense_counts = [5, 10, 15]
            n_pairs = dense_counts[index % len(dense_counts)]

            sampled = self._sample_pairs(pool, n_pairs, partition)
            target_pair = sampled[0]
            distractors = sampled[1:]
            all_pairs = [target_pair] + distractors
            self.rng.shuffle(all_pairs)

            (q_id, q_str), (tgt_id, tgt_str) = target_pair
            body = ", ".join([f"{k}={v}" for (k_id, k), (v_id, v) in all_pairs])
            prompt = f"DENSE_PAIRS: {body}. QUERY: {q_str} ->"

            return DatasetRecord(
                record_id=f"{partition.value}_{task_family.value}_{index:05d}",
                task_family=task_family.value,
                partition=partition.value,
                prompt_text=prompt,
                target_token=tgt_str,
                target_token_id=tgt_id,
                context_pairs=[(k, v) for (k_id, k), (v_id, v) in all_pairs],
                context_pair_ids=[(k_id, v_id) for (k_id, k), (v_id, v) in all_pairs],
                query_key=q_str,
                query_key_id=q_id,
                distractor_count=len(distractors),
                key_val_distance=2,
                prompt_length_tokens=len(prompt.split()),
            )

        raise ValueError(f"Unknown Task Family: {task_family}")

    def generate_partition(
        self,
        partition: PartitionType,
        counts_per_family: Dict[TaskFamily, int],
    ) -> List[DatasetRecord]:
        """Generates all records for a partition with strict integrity checking."""
        records: List[DatasetRecord] = []
        for tf, count in counts_per_family.items():
            for i in range(count):
                rec = self.generate_record(tf, partition, i)
                # Check for duplicate prompt hash
                assert rec.prompt_sha256 not in self.prompt_hashes[partition.value], (
                    f"Duplicate prompt hash detected in partition {partition.value} for record {rec.record_id}!"
                )
                self.prompt_hashes[partition.value].add(rec.prompt_sha256)
                records.append(rec)
        return records

    def verify_all_isolation_invariants(self) -> Dict[str, Any]:
        """
        Audits all mathematical isolation invariants across generated splits:
        1. Token pool disjointness (E_train vs E_test)
        2. Bidirectional association leakage (VER-03)
        3. Cross-partition prompt hash collisions
        """
        # 1. Token disjointness
        e_train_ids = self.pool.train_token_ids
        e_test_ids = self.pool.test_token_ids
        token_overlap = e_train_ids.intersection(e_test_ids)

        # 2. Association overlap & bidirectional exclusion
        train_sym = self.train_symmetric_closure
        val_pairs = self.seen_pairs[PartitionType.PARTITION_B_VAL_RECOMB.value]
        test_pairs = self.seen_pairs[PartitionType.PARTITION_C_TEST_RECOMB.value]
        disjoint_pairs = self.seen_pairs[PartitionType.PARTITION_D_TEST_DISJOINT.value]

        val_leakage = train_sym.intersection(val_pairs)
        test_leakage = train_sym.intersection(test_pairs)
        disjoint_leakage = train_sym.intersection(disjoint_pairs)

        # 3. Prompt hash collisions across partitions
        hashes_a = self.prompt_hashes[PartitionType.PARTITION_A_TRAIN.value]
        hashes_b = self.prompt_hashes[PartitionType.PARTITION_B_VAL_RECOMB.value]
        hashes_c = self.prompt_hashes[PartitionType.PARTITION_C_TEST_RECOMB.value]
        hashes_d = self.prompt_hashes[PartitionType.PARTITION_D_TEST_DISJOINT.value]

        cross_hash_collisions = (
            len(hashes_a.intersection(hashes_b)) +
            len(hashes_a.intersection(hashes_c)) +
            len(hashes_a.intersection(hashes_d)) +
            len(hashes_b.intersection(hashes_c))
        )

        audit_passed = (
            len(token_overlap) == 0 and
            len(val_leakage) == 0 and
            len(test_leakage) == 0 and
            len(disjoint_leakage) == 0 and
            cross_hash_collisions == 0
        )

        return {
            "audit_passed": audit_passed,
            "token_disjointness_passed": len(token_overlap) == 0,
            "token_overlap_count": len(token_overlap),
            "val_symmetric_leakage_count": len(val_leakage),
            "test_symmetric_leakage_count": len(test_leakage),
            "disjoint_leakage_count": len(disjoint_leakage),
            "cross_partition_hash_collisions": cross_hash_collisions,
            "total_train_symmetric_associations": len(train_sym),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="CAND_01 Synthetic Dataset Generator")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Run in synthetic mode (default: True)")
    parser.add_argument("--output-dir", type=str, default="data/fixtures/cand01_synthetic", help="Output directory for synthetic fixtures")
    parser.add_argument("--scale", type=str, default="mock", choices=["mock", "full_scale_synthetic"], help="Dataset size scale")
    parser.add_argument("--seed", type=int, default=999042, help="Deterministic generator seed")
    parser.add_argument("--production", action="store_true", help="FORBIDDEN: Attempt to generate production data")
    return parser.parse_args()


def main():
    args = parse_args()

    # Hard boundary check: production dataset generation is strictly unauthorized
    abs_out = os.path.abspath(args.output_dir).lower()
    norm_forbidden = os.path.normpath("data/experiments/cand_01").lower()
    if args.production or (norm_forbidden in abs_out and "fixtures" not in abs_out):
        sys.stderr.write(
            "FATAL GOVERNANCE BREACH: Production dataset generation is strictly UNAUTHORIZED.\n"
            "This script is restricted to synthetic fixtures in non-production directories.\n"
        )
        sys.exit(1)

    print("=================================================================")
    print("CAND_01 SYNTHETIC DATASET GENERATION (SAFE ISOLATED MODE)")
    print(f"Seed: {args.seed} | Scale: {args.scale} | Output: {args.output_dir}")
    print("=================================================================")

    pool = SyntheticEntityPool(n_train=450, n_test=150, seed=args.seed)
    gen = CAND01DatasetGenerator(pool, seed=args.seed)

    if args.scale == "mock":
        # Comprehensive mock counts across all 8 Task Families
        train_counts = {
            TaskFamily.TF01_DIRECT_COPY: 10,
            TaskFamily.TF02_SINGLE_PAIR: 15,
            TaskFamily.TF03_MULTI_PAIR: 15,
            TaskFamily.TF04_DISTRACTOR: 15,
            TaskFamily.TF05_SIMILAR_KEY: 15,
        }
        val_counts = {
            TaskFamily.TF02_SINGLE_PAIR: 5,
            TaskFamily.TF03_MULTI_PAIR: 5,
            TaskFamily.TF04_DISTRACTOR: 5,
            TaskFamily.TF05_SIMILAR_KEY: 5,
            TaskFamily.TF06_RECOMBINATION: 5,
            TaskFamily.TF07_OFFSET_SCALING: 5,
            TaskFamily.TF08_CONTEXT_SCALING: 5,
        }
        test_counts = {
            TaskFamily.TF02_SINGLE_PAIR: 5,
            TaskFamily.TF03_MULTI_PAIR: 5,
            TaskFamily.TF04_DISTRACTOR: 5,
            TaskFamily.TF05_SIMILAR_KEY: 5,
            TaskFamily.TF06_RECOMBINATION: 5,
            TaskFamily.TF07_OFFSET_SCALING: 5,
            TaskFamily.TF08_CONTEXT_SCALING: 5,
        }
        disjoint_counts = {
            TaskFamily.TF01_DIRECT_COPY: 5,
            TaskFamily.TF02_SINGLE_PAIR: 5,
            TaskFamily.TF03_MULTI_PAIR: 5,
            TaskFamily.TF04_DISTRACTOR: 5,
            TaskFamily.TF05_SIMILAR_KEY: 5,
            TaskFamily.TF06_RECOMBINATION: 5,
            TaskFamily.TF07_OFFSET_SCALING: 5,
            TaskFamily.TF08_CONTEXT_SCALING: 5,
        }
    else:
        # Full scale synthetic (4000 / 500 / 500 / 500)
        train_counts = {
            TaskFamily.TF01_DIRECT_COPY: 350,
            TaskFamily.TF02_SINGLE_PAIR: 913,
            TaskFamily.TF03_MULTI_PAIR: 913,
            TaskFamily.TF04_DISTRACTOR: 912,
            TaskFamily.TF05_SIMILAR_KEY: 912,
        }
        val_counts = {
            TaskFamily.TF02_SINGLE_PAIR: 71,
            TaskFamily.TF03_MULTI_PAIR: 71,
            TaskFamily.TF04_DISTRACTOR: 71,
            TaskFamily.TF05_SIMILAR_KEY: 71,
            TaskFamily.TF06_RECOMBINATION: 72,
            TaskFamily.TF07_OFFSET_SCALING: 72,
            TaskFamily.TF08_CONTEXT_SCALING: 72,
        }
        test_counts = {
            TaskFamily.TF02_SINGLE_PAIR: 71,
            TaskFamily.TF03_MULTI_PAIR: 71,
            TaskFamily.TF04_DISTRACTOR: 71,
            TaskFamily.TF05_SIMILAR_KEY: 71,
            TaskFamily.TF06_RECOMBINATION: 72,
            TaskFamily.TF07_OFFSET_SCALING: 72,
            TaskFamily.TF08_CONTEXT_SCALING: 72,
        }
        disjoint_counts = {
            TaskFamily.TF01_DIRECT_COPY: 50,
            TaskFamily.TF02_SINGLE_PAIR: 64,
            TaskFamily.TF03_MULTI_PAIR: 64,
            TaskFamily.TF04_DISTRACTOR: 64,
            TaskFamily.TF05_SIMILAR_KEY: 64,
            TaskFamily.TF06_RECOMBINATION: 65,
            TaskFamily.TF07_OFFSET_SCALING: 65,
            TaskFamily.TF08_CONTEXT_SCALING: 64,
        }

    train_recs = gen.generate_partition(PartitionType.PARTITION_A_TRAIN, train_counts)
    val_recs = gen.generate_partition(PartitionType.PARTITION_B_VAL_RECOMB, val_counts)
    test_recs = gen.generate_partition(PartitionType.PARTITION_C_TEST_RECOMB, test_counts)
    disjoint_recs = gen.generate_partition(PartitionType.PARTITION_D_TEST_DISJOINT, disjoint_counts)

    print(f"Generated records:")
    print(f"  Train: {len(train_recs)}")
    print(f"  Val Recomb: {len(val_recs)}")
    print(f"  Test Recomb: {len(test_recs)}")
    print(f"  Test Disjoint: {len(disjoint_recs)}")

    audit = gen.verify_all_isolation_invariants()
    print("\nIsolation & Leakage Invariant Audit:")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    if not audit["audit_passed"]:
        sys.stderr.write("FATAL: Isolation audit failed!\n")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, recs in [
        ("train.jsonl", train_recs),
        ("val_recomb.jsonl", val_recs),
        ("test_recomb.jsonl", test_recs),
        ("test_disjoint.jsonl", disjoint_recs),
    ]:
        out_path = os.path.join(args.output_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"Wrote synthetic fixture: {out_path} ({len(recs)} records)")

    print("\nSUCCESS: Dataset generator executed cleanly in isolated synthetic mode.")


if __name__ == "__main__":
    main()
