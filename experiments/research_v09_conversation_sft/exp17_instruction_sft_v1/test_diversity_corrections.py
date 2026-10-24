"""
test_diversity_corrections.py
=============================
Focused test suite for Exp17 dataset diversity corrections.
Verifies candidate banks and generator consistency without writing official split files.
"""

import unittest
import json
import difflib
from pathlib import Path

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp17_instruction_sft_v1.candidate_banks import (
    CANDIDATE_FACTUAL_BANKS, CANDIDATE_EXPLANATION_BANKS, CANDIDATE_ABSTENTION_BANKS
)
from experiments.research_v09_conversation_sft.exp17_instruction_sft_v1.generate_exp17_datasets import (
    Exp17DatasetGenerator, TOKENIZER_DIR, FORBIDDEN_GATE_ENTITIES, QUOTAS, CATEGORIES
)

class TestExp17DiversityCorrections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.tok_cfg = load_tokenizer_artifacts(TOKENIZER_DIR, verify_checksums=True)
        gate_path = Path(__file__).resolve().parents[2] / "normal_chat_gate_v01" / "prompts.jsonl"
        with open(gate_path, "r", encoding="utf-8") as f:
            cls.gate_prompts = [json.loads(line) for line in f]

    def test_quotas_and_split_allocation(self):
        """Verify candidate banks exactly fulfill the 40/5/5 allocation per split."""
        for split, count in [("train", 40), ("val", 5), ("test", 5)]:
            self.assertEqual(len(CANDIDATE_FACTUAL_BANKS[split]), count)
            self.assertEqual(len(CANDIDATE_EXPLANATION_BANKS[split]), count)
            self.assertEqual(len(CANDIDATE_ABSTENTION_BANKS[split]), count)

    def test_factual_bank_uniqueness_and_token_bounds(self):
        """Verify all 50 factual QA items are unique and bounded within [3, 45] tokens."""
        all_q = set()
        all_a = set()
        for split, items in CANDIDATE_FACTUAL_BANKS.items():
            for q, a, intent, note in items:
                self.assertNotIn(q.lower().strip(), all_q, f"Duplicate factual question: {q}")
                self.assertNotIn(a.lower().strip(), all_a, f"Duplicate factual answer: {a}")
                all_q.add(q.lower().strip())
                all_a.add(a.lower().strip())

                tokens = self.tokenizer.encode(a.strip())
                self.assertGreaterEqual(len(tokens), 3)
                self.assertLessEqual(len(tokens), 45)

    def test_explanation_bank_uniqueness_and_mechanisms(self):
        """Verify all 50 scientific explanations are unique and bounded within [3, 45] tokens."""
        all_q = set()
        all_a = set()
        all_mechs = set()
        for split, items in CANDIDATE_EXPLANATION_BANKS.items():
            for q, a, intent, mech in items:
                self.assertNotIn(q.lower().strip(), all_q, f"Duplicate explanation question: {q}")
                self.assertNotIn(a.lower().strip(), all_a, f"Duplicate explanation answer: {a}")
                all_q.add(q.lower().strip())
                all_a.add(a.lower().strip())
                all_mechs.add(mech.lower().strip())

                tokens = self.tokenizer.encode(a.strip())
                self.assertGreaterEqual(len(tokens), 3)
                self.assertLessEqual(len(tokens), 45)
        self.assertEqual(len(all_mechs), 50, "Expected 50 distinct scientific mechanism concepts.")

    def test_abstention_domains_and_controls(self):
        """Verify abstention bank has >=6 distinct domains, pairs controls, and respects token limits."""
        domains = set()
        all_unans_a = []
        for split, items in CANDIDATE_ABSTENTION_BANKS.items():
            for ctx, uq, ua, cq, ca, domain in items:
                domains.add(domain)
                all_unans_a.append(ua.strip())
                # Verify control query can be answered from context
                cq_tokens = set(cq.lower().replace("?", "").split())
                ca_tokens = set(ca.lower().replace(".", "").split())
                ctx_tokens = set(ctx.lower().replace(".", "").split())
                # Context must contain key answer tokens for the control
                intersection = ca_tokens.intersection(ctx_tokens)
                self.assertGreater(len(intersection), 0, f"Control answer '{ca}' not grounded in context '{ctx}'")

                # Verify token lengths
                u_tok = self.tokenizer.encode(ua.strip())
                c_tok = self.tokenizer.encode(ca.strip())
                self.assertGreaterEqual(len(u_tok), 3)
                self.assertLessEqual(len(u_tok), 45)
                self.assertGreaterEqual(len(c_tok), 3)
                self.assertLessEqual(len(c_tok), 45)

        self.assertGreaterEqual(len(domains), 6, f"Expected >= 6 domains, got {len(domains)}")
        # Check phrase reuse frequency in Train: max frequency <= 6
        train_ua = [ua for ctx, uq, ua, cq, ca, d in CANDIDATE_ABSTENTION_BANKS["train"]]
        import collections
        counts = collections.Counter(train_ua)
        self.assertLessEqual(max(counts.values()), 6, f"Max abstention phrase reuse exceeds 6: {counts.most_common(1)}")

    def test_no_contamination_with_gate_prompts(self):
        """Verify zero gate entity blacklist hits and similarity < 0.70 across all candidate queries."""
        all_queries = []
        for split, items in CANDIDATE_FACTUAL_BANKS.items():
            for q, a, intent, note in items:
                all_queries.append(q)
        for split, items in CANDIDATE_EXPLANATION_BANKS.items():
            for q, a, intent, note in items:
                all_queries.append(q)
        for split, items in CANDIDATE_ABSTENTION_BANKS.items():
            for ctx, uq, ua, cq, ca, d in items:
                all_queries.append(uq)
                all_queries.append(cq)

        for q in all_queries:
            q_lower = q.lower().strip()
            for ent in FORBIDDEN_GATE_ENTITIES:
                self.assertNotIn(ent, q_lower, f"Forbidden gate entity '{ent}' in candidate query: {q}")
            for gp in self.gate_prompts:
                ratio = difflib.SequenceMatcher(None, q_lower, gp["prompt"].lower().strip()).ratio()
                self.assertLess(ratio, 0.70, f"Similarity {ratio:.3f} >= 0.70 with Gate prompt {gp['id']}: {q}")

    def test_generator_integrated_output_contains_candidate_banks(self):
        """Verify that the actual generator in-memory output uses the candidate banks."""
        gen = Exp17DatasetGenerator()
        dataset = gen.generate_all_splits()

        # Check total record count and split allocation
        self.assertEqual(len(dataset["train"]), 320)
        self.assertEqual(len(dataset["val"]), 40)
        self.assertEqual(len(dataset["test"]), 40)

        # 1. Verify Factual QA in actual generated dataset
        for split in ["train", "val", "test"]:
            gen_fqa = [r for r in dataset[split] if r["category"] == "factual_qa"]
            bank_fqa = CANDIDATE_FACTUAL_BANKS[split]
            self.assertEqual(len(gen_fqa), len(bank_fqa))
            for i, r in enumerate(gen_fqa):
                expected_q, expected_a, _, _ = bank_fqa[i]
                self.assertEqual(r["prompt"], expected_q)
                self.assertEqual(r["response"], expected_a)

        # 2. Verify Scientific Explanations in actual generated dataset
        for split in ["train", "val", "test"]:
            gen_exp = [r for r in dataset[split] if r["category"] == "short_explanation"]
            bank_exp = CANDIDATE_EXPLANATION_BANKS[split]
            self.assertEqual(len(gen_exp), len(bank_exp))
            for i, r in enumerate(gen_exp):
                expected_q, expected_a, _, _ = bank_exp[i]
                self.assertEqual(r["prompt"], expected_q)
                self.assertEqual(r["response"], expected_a)

        # 3. Verify Uncertainty & Abstention (including controls) in actual generated dataset
        for split in ["train", "val", "test"]:
            gen_abst = [r for r in dataset[split] if r["category"] == "uncertainty_abstention"]
            bank_abst = CANDIDATE_ABSTENTION_BANKS[split]
            self.assertEqual(len(gen_abst), len(bank_abst))
            
            # Check answerable controls exist and are marked correctly
            controls = [r for r in gen_abst if not r["metadata"]["requires_abstention"]]
            unanswerable = [r for r in gen_abst if r["metadata"]["requires_abstention"]]
            expected_controls_count = 10 if split == "train" else 1
            expected_unans_count = 30 if split == "train" else 4
            self.assertEqual(len(controls), expected_controls_count)
            self.assertEqual(len(unanswerable), expected_unans_count)

    def test_generator_deterministic_reproducibility(self):
        """Verify two independent runs of Exp17DatasetGenerator produce bitwise identical datasets."""
        gen1 = Exp17DatasetGenerator()
        ds1 = gen1.generate_all_splits()

        gen2 = Exp17DatasetGenerator()
        ds2 = gen2.generate_all_splits()

        s1 = json.dumps(ds1, sort_keys=True)
        s2 = json.dumps(ds2, sort_keys=True)
        self.assertEqual(s1, s2, "Generator output must be bitwise identical across independent runs with same seeds.")

    def test_generator_token_bounds_and_target_distribution(self):
        """Verify all 400 generated records comply with token bounds and target distribution."""
        gen = Exp17DatasetGenerator()
        dataset = gen.generate_all_splits()
        all_recs = dataset["train"] + dataset["val"] + dataset["test"]

        in_target_count = 0
        for r in all_recs:
            tokens = self.tokenizer.encode(r["response"].strip())
            tok_len = len(tokens)
            self.assertEqual(tok_len, r["metadata"]["target_token_count"])
            self.assertGreaterEqual(tok_len, 3)
            self.assertLessEqual(tok_len, 45)
            if 5 <= tok_len <= 35:
                in_target_count += 1

        pct_target = (in_target_count / len(all_recs)) * 100
        self.assertGreaterEqual(pct_target, 90.0, f"Expected >= 90% in 5-35 tokens, got {pct_target:.1f}%")

    def test_generator_cross_split_disjointness(self):
        """Verify zero prompt overlap across Train, Val, and Test in actual generator output."""
        gen = Exp17DatasetGenerator()
        dataset = gen.generate_all_splits()

        tr_p = set(r["prompt"].strip().lower() for r in dataset["train"])
        va_p = set(r["prompt"].strip().lower() for r in dataset["val"])
        te_p = set(r["prompt"].strip().lower() for r in dataset["test"])

        self.assertEqual(len(tr_p.intersection(va_p)), 0, "Train-Val prompt overlap detected!")
        self.assertEqual(len(tr_p.intersection(te_p)), 0, "Train-Test prompt overlap detected!")
        self.assertEqual(len(va_p.intersection(te_p)), 0, "Val-Test prompt overlap detected!")

if __name__ == "__main__":
    unittest.main()

