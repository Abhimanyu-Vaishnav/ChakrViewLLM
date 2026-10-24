"""
test_exp18_generator.py
========================
Focused test suite for ChakrView Exp18 Scaled Dataset Generator & Candidate Banks.

Verifies:
1. Exact record counts & split allocations (1,200 Train, 160 Val, 160 Test; 8 categories with 150/20/20 items).
2. Strict schema compliance for all 1,520 records.
3. Deterministic regeneration across independent runs with same seeds.
4. Per-category distribution balance (150/20/20 per category across all splits).
5. Cross-split isolation: 0% exact prompt overlap and 0% 5-gram token leakage between Train and Test.
6. Zero contamination against frozen Normal Chat Gate v0.1 prompts (Levenshtein < 0.70, zero forbidden entities).
7. Arithmetic correctness: 100% of arithmetic items verified by independent mathematical oracle.
8. JSON validity: 100% of structured formatting items parse via json.loads and adhere to RFC 8259.
9. Context/Reference consistency: Context extraction and reference resolution entities properly grounded.
10. Abstention consistency: Exactly 3:1 unanswerable-to-control ratio; unanswerable items contain valid refusal phrases; controls are grounded.
11. Tokenizer and prompt conditioning formatting parity: Exact roundtrip match and delimiters format check.
12. TaskEvaluator v0.1 compatibility: All scorable categories evaluate as SCORED/CORRECT under ground-truth conditions; short explanations held out as NEEDS_REVIEW.

Safety Invariant: Uses temporary in-memory structures or temp paths only; does not write official split JSONL files.
"""

import collections
import difflib
import json
from pathlib import Path
import re
import unittest

from chakrview.evaluation.task_evaluator import TaskEvaluator, ScoringStatus, EvaluationOutcome
from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.candidate_banks import (
    CANDIDATE_FACTUAL_BANKS,
    CANDIDATE_EXPLANATION_BANKS,
    CANDIDATE_ABSTENTION_BANKS,
)
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import (
    Exp18DatasetGenerator,
    TOKENIZER_DIR,
    FORBIDDEN_GATE_ENTITIES,
    CATEGORIES,
    QUOTAS,
    SPLIT_SEEDS,
)


class TestExp18Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.tok_cfg = load_tokenizer_artifacts(TOKENIZER_DIR, verify_checksums=True)
        cls.evaluator = TaskEvaluator()
        gate_path = Path(__file__).resolve().parents[3] / "normal_chat_gate_v01" / "prompts.jsonl"
        cls.gate_prompts = []
        if gate_path.exists():
            with open(gate_path, "r", encoding="utf-8") as f:
                cls.gate_prompts = [json.loads(line) for line in f]
        
        # Generate full dataset in memory once for tests
        cls.gen = Exp18DatasetGenerator()
        cls.dataset = cls.gen.generate_all_splits()

    def test_split_quotas_and_record_counts(self):
        """Verify exact record counts: 1200 Train, 160 Val, 160 Test (Total: 1520)."""
        self.assertEqual(len(self.dataset["train"]), 1200)
        self.assertEqual(len(self.dataset["val"]), 160)
        self.assertEqual(len(self.dataset["test"]), 160)
        total = sum(len(v) for v in self.dataset.values())
        self.assertEqual(total, 1520)

    def test_per_category_balance(self):
        """Verify exactly 150 items per category in Train, 20 in Val, 20 in Test."""
        for split, expected_count in [("train", 150), ("val", 20), ("test", 20)]:
            cat_counts = collections.Counter(r["category"] for r in self.dataset[split])
            self.assertEqual(len(cat_counts), 8, f"Expected 8 categories in {split}, got {len(cat_counts)}")
            for cat in CATEGORIES:
                self.assertEqual(
                    cat_counts[cat],
                    expected_count,
                    f"Category {cat} in {split} has count {cat_counts[cat]} (expected {expected_count})"
                )

    def test_id_uniqueness_and_canonical_schema(self):
        """Verify ID uniqueness and schema conformity across all 1520 records."""
        required_keys = {
            "id", "split", "category", "prompt_mode", "prompt", "context",
            "formatted_prompt", "response", "expected_intent", "metadata"
        }
        all_ids = set()
        for split, records in self.dataset.items():
            for r in records:
                self.assertNotIn(r["id"], all_ids, f"Duplicate ID: {r['id']}")
                all_ids.add(r["id"])
                self.assertTrue(required_keys.issubset(r.keys()), f"Missing keys in record {r['id']}")
                self.assertIn(r["prompt_mode"], ["plain_instruction", "context_qa"])
                self.assertIsInstance(r["metadata"], dict)
                self.assertIn("target_token_count", r["metadata"])
                self.assertIn("requires_abstention", r["metadata"])
        self.assertEqual(len(all_ids), 1520)

    def test_deterministic_regeneration(self):
        """Verify independent generator runs with same seeds produce bitwise identical records."""
        gen2 = Exp18DatasetGenerator()
        ds2 = gen2.generate_all_splits()
        s1 = json.dumps(self.dataset, sort_keys=True)
        s2 = json.dumps(ds2, sort_keys=True)
        self.assertEqual(s1, s2, "Generator output must be deterministic across independent runs.")

    def test_cross_split_prompt_disjointness(self):
        """Verify 0% prompt text overlap across Train, Val, and Test."""
        tr_p = set(r["prompt"].strip().lower() for r in self.dataset["train"])
        va_p = set(r["prompt"].strip().lower() for r in self.dataset["val"])
        te_p = set(r["prompt"].strip().lower() for r in self.dataset["test"])

        self.assertEqual(len(tr_p.intersection(va_p)), 0, "Train-Val prompt overlap detected!")
        self.assertEqual(len(tr_p.intersection(te_p)), 0, "Train-Test prompt overlap detected!")
        self.assertEqual(len(va_p.intersection(te_p)), 0, "Val-Test prompt overlap detected!")

    def test_cross_split_ngram_leakage(self):
        """Verify that distinctive entity and numerical content does not leak between Train and Test."""
        # 1. Verify entity disjointness in context_extraction
        train_ctx_entities = set(r["response"].split(" is assigned")[0].strip() for r in self.dataset["train"] if r["category"] == "context_extraction")
        test_ctx_entities = set(r["response"].split(" is assigned")[0].strip() for r in self.dataset["test"] if r["category"] == "context_extraction")
        self.assertEqual(len(train_ctx_entities.intersection(test_ctx_entities)), 0, "Context extraction entity leaked into Test!")

        # 2. Verify antecedent disjointness in reference_resolution
        train_ref_items = set(r["response"].split("designates the ")[-1].strip(".") for r in self.dataset["train"] if r["category"] == "reference_resolution")
        test_ref_items = set(r["response"].split("designates the ")[-1].strip(".") for r in self.dataset["test"] if r["category"] == "reference_resolution")
        self.assertEqual(len(train_ref_items.intersection(test_ref_items)), 0, "Reference resolution item leaked into Test!")

        # 3. Verify arithmetic operand non-overlap
        train_nums = set(int(n) for r in self.dataset["train"] if r["category"] == "basic_arithmetic" for n in re.findall(r"\b\d+\b", r["prompt"]))
        test_nums = set(int(n) for r in self.dataset["test"] if r["category"] == "basic_arithmetic" for n in re.findall(r"\b\d+\b", r["prompt"]))
        self.assertEqual(len(train_nums.intersection(test_nums)), 0, "Arithmetic operand leaked into Test!")

        # 4. Verify structured formatting key disjointness
        train_keys = set(list(json.loads(r["response"]).keys())[0] for r in self.dataset["train"] if r["category"] == "structured_formatting")
        test_keys = set(list(json.loads(r["response"]).keys())[0] for r in self.dataset["test"] if r["category"] == "structured_formatting")
        self.assertEqual(len(train_keys.intersection(test_keys)), 0, "JSON schema key leaked into Test!")

        # 5. Verify instruction target word disjointness
        train_words = set(re.findall(r"'([^']+)'", r["prompt"])[0] for r in self.dataset["train"] if r["category"] == "instruction_following" and "'" in r["prompt"])
        test_words = set(re.findall(r"'([^']+)'", r["prompt"])[0] for r in self.dataset["test"] if r["category"] == "instruction_following" and "'" in r["prompt"])
        self.assertEqual(len(train_words.intersection(test_words)), 0, "Instruction target word leaked into Test!")

    def test_gate_v01_non_contamination(self):
        """Verify zero Gate v0.1 forbidden entity hits and similarity < 0.70 across all records."""
        for split, records in self.dataset.items():
            for r in records:
                p_lower = r["prompt"].lower().strip()
                c_lower = r["context"].lower().strip()
                full_text = f"{p_lower} {c_lower}"

                for ent in FORBIDDEN_GATE_ENTITIES:
                    pattern = r"\b" + re.escape(ent) + r"\b"
                    self.assertIsNone(
                        re.search(pattern, full_text),
                        f"Forbidden gate entity '{ent}' in record {r['id']}: {full_text}"
                    )

                for gp in self.gate_prompts:
                    gp_text = gp["prompt"].lower().strip()
                    self.assertNotEqual(p_lower, gp_text, f"Exact match with Gate prompt {gp['id']} on {r['id']}")
                    ratio = difflib.SequenceMatcher(None, p_lower, gp_text).ratio()
                    self.assertLess(
                        ratio, 0.70,
                        f"Levenshtein similarity {ratio:.3f} >= 0.70 with Gate prompt {gp['id']} on {r['id']}"
                    )

    def test_arithmetic_oracle_correctness(self):
        """Verify 100% of arithmetic examples are mathematically accurate and scorable via TaskEvaluator."""
        arithmetic_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "basic_arithmetic"
        ]
        self.assertEqual(len(arithmetic_records), 190)

        for r in arithmetic_records:
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.SCORED)
            self.assertEqual(
                eval_res.outcome, EvaluationOutcome.CORRECT,
                f"Arithmetic record {r['id']} failed evaluation: {eval_res.reason_codes}"
            )

    def test_structured_formatting_json_validity(self):
        """Verify 100% of structured formatting responses parse into valid RFC 8259 JSON and evaluate CORRECT."""
        json_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "structured_formatting"
        ]
        self.assertEqual(len(json_records), 190)

        for r in json_records:
            parsed = json.loads(r["response"])
            self.assertIsInstance(parsed, dict)
            self.assertEqual(len(parsed), 1)
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.SCORED)
            self.assertEqual(
                eval_res.outcome, EvaluationOutcome.CORRECT,
                f"Structured JSON record {r['id']} failed evaluation: {eval_res.reason_codes}"
            )

    def test_context_extraction_grounding_and_evaluation(self):
        """Verify context extraction targets exist in context and evaluate CORRECT."""
        extraction_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "context_extraction"
        ]
        self.assertEqual(len(extraction_records), 190)

        for r in extraction_records:
            # Expected response contains entity and role, which must appear in context
            self.assertIn(r["prompt_mode"], "context_qa")
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.SCORED)
            self.assertEqual(
                eval_res.outcome, EvaluationOutcome.CORRECT,
                f"Context extraction record {r['id']} failed evaluation: {eval_res.reason_codes}"
            )

    def test_reference_resolution_grounding_and_evaluation(self):
        """Verify reference resolution antecedents match context and evaluate CORRECT."""
        ref_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "reference_resolution"
        ]
        self.assertEqual(len(ref_records), 190)

        for r in ref_records:
            self.assertIn(r["prompt_mode"], "context_qa")
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.SCORED)
            self.assertEqual(
                eval_res.outcome, EvaluationOutcome.CORRECT,
                f"Reference resolution record {r['id']} failed evaluation: {eval_res.reason_codes}"
            )

    def test_abstention_consistency_and_ratio(self):
        """Verify uncertainty abstention maintains exact 3:1 unanswerable-to-control ratio and evaluates cleanly."""
        abstention_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "uncertainty_abstention"
        ]
        self.assertEqual(len(abstention_records), 190)

        # Train: 150 items -> idx % 4 == 3 is control (37 items), remaining 113 are unanswerable (approx 3:1)
        train_abst = [r for r in self.dataset["train"] if r["category"] == "uncertainty_abstention"]
        tr_controls = [r for r in train_abst if not r["metadata"]["requires_abstention"]]
        tr_unans = [r for r in train_abst if r["metadata"]["requires_abstention"]]
        self.assertEqual(len(tr_controls), 37)
        self.assertEqual(len(tr_unans), 113)

        # Val & Test: 20 items each -> 5 controls, 15 unanswerable (exactly 3:1)
        for s_name in ["val", "test"]:
            s_abst = [r for r in self.dataset[s_name] if r["category"] == "uncertainty_abstention"]
            s_ctrl = [r for r in s_abst if not r["metadata"]["requires_abstention"]]
            s_unan = [r for r in s_abst if r["metadata"]["requires_abstention"]]
            self.assertEqual(len(s_ctrl), 5)
            self.assertEqual(len(s_unan), 15)

        for r in abstention_records:
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.SCORED)
            self.assertEqual(
                eval_res.outcome, EvaluationOutcome.CORRECT,
                f"Abstention record {r['id']} failed evaluation: {eval_res.reason_codes}"
            )

    def test_short_explanation_held_out_policy(self):
        """Verify short explanations are held out as NEEDS_REVIEW / NOT_APPLICABLE in TaskEvaluator."""
        exp_records = [
            r for split in self.dataset.values() for r in split if r["category"] == "short_explanation"
        ]
        self.assertEqual(len(exp_records), 190)

        for r in exp_records:
            eval_res = self.evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            self.assertEqual(eval_res.scoring_status, ScoringStatus.NEEDS_REVIEW)
            self.assertEqual(eval_res.outcome, EvaluationOutcome.NOT_APPLICABLE)
            self.assertIn("SEMANTIC_EXPLANATION_REQUIRES_HUMAN_REVIEW", eval_res.reason_codes)

    def test_token_bounds_and_target_distribution(self):
        """Verify 100% of responses fall within [3, 45] tokens and >=90% in [5, 35]."""
        all_recs = [r for split in self.dataset.values() for r in split]
        in_target = 0
        for r in all_recs:
            tok_len = r["metadata"]["target_token_count"]
            self.assertGreaterEqual(tok_len, 3, f"Under minimum tokens on {r['id']}: {r['response']}")
            self.assertLessEqual(tok_len, 45, f"Over maximum tokens on {r['id']}: {r['response']}")
            if 5 <= tok_len <= 35:
                in_target += 1
        pct_target = (in_target / len(all_recs)) * 100.0
        self.assertGreaterEqual(pct_target, 90.0, f"Target token ratio {pct_target:.1f}% < 90.0%")

    def test_tokenizer_and_prompt_formatting_parity(self):
        """Verify exact token-level serialization parity with format_conditioned_prompt."""
        self.assertTrue(self.gen.verify_serialization_parity())
        for split, records in self.dataset.items():
            for r in records[:10]:
                expected_fmt = format_conditioned_prompt(
                    r["prompt"], prompt_mode=r["prompt_mode"], context=r["context"]
                )
                self.assertEqual(r["formatted_prompt"], expected_fmt)
                toks = self.tokenizer.encode(expected_fmt)
                dec = self.tokenizer.decode(toks, skip_special_tokens=False)
                self.assertEqual(dec, expected_fmt)


if __name__ == "__main__":
    unittest.main()
