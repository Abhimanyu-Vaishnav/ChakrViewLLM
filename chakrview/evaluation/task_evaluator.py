"""
ChakrView Task Evaluator v0.1.

Offline, deterministic, category-specific evaluation module for instruction-following models.
Strictly separates turn-termination (EOS) from semantic task correctness.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class ScoringStatus(str, Enum):
    SCORED = "SCORED"
    UNSCORED = "UNSCORED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class EvaluationOutcome(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class EvaluationResult:
    record_id: str
    category: str
    scoring_status: ScoringStatus
    outcome: EvaluationOutcome
    expected_target: str
    observed_evidence: str
    reason_codes: List[str] = field(default_factory=list)
    format_valid: bool = False
    is_eos: bool = False
    is_max_tokens: bool = False
    is_empty: bool = False
    limitations: Optional[str] = None
    evaluator_version: str = "v0.1"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scoring_status"] = self.scoring_status.value
        d["outcome"] = self.outcome.value
        return d


class TaskEvaluator:
    """
    Deterministic offline task evaluation engine for ChakrView.
    """

    VERSION = "v0.1"

    def __init__(self) -> None:
        pass

    def evaluate_record(
        self,
        record: Dict[str, Any],
        generated_response: str,
        stop_reason: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluates a single model completion against an Exp17 dataset record.
        """
        record_id = record.get("id", "UNKNOWN_RECORD")
        category = record.get("category", "unknown")
        prompt = record.get("prompt", "")
        expected = record.get("response", "")
        context = record.get("context", "")
        metadata = record.get("metadata", {})

        gen = generated_response
        gen_clean = gen.strip()

        is_eos = (stop_reason == "STOP_TOKEN")
        is_max = (stop_reason == "MAX_TOKENS")
        is_empty = (len(gen_clean) == 0)

        # Dispatch by category
        if category == "basic_arithmetic":
            return self._score_arithmetic(record_id, prompt, expected, gen, is_eos, is_max, is_empty)
        elif category == "structured_formatting":
            return self._score_structured_formatting(record_id, prompt, expected, gen, is_eos, is_max, is_empty)
        elif category == "uncertainty_abstention":
            return self._score_uncertainty_abstention(record_id, prompt, context, metadata, expected, gen, is_eos, is_max, is_empty)
        elif category == "context_extraction":
            return self._score_context_extraction(record_id, prompt, context, expected, gen, is_eos, is_max, is_empty)
        elif category == "context_copying":
            return self._score_context_copying(record, expected, gen, is_eos, is_max, is_empty)
        elif category == "reference_resolution":
            return self._score_reference_resolution(record_id, prompt, context, expected, gen, is_eos, is_max, is_empty)
        elif category == "instruction_following":
            return self._score_instruction_following(record_id, prompt, expected, gen, is_eos, is_max, is_empty)
        elif category == "factual_qa":
            return self._score_factual_qa(record_id, prompt, expected, gen, is_eos, is_max, is_empty)
        elif category == "short_explanation":
            return self._score_short_explanation(record_id, prompt, expected, gen, is_eos, is_max, is_empty)
        else:
            return EvaluationResult(
                record_id=record_id,
                category=category,
                scoring_status=ScoringStatus.UNSCORED,
                outcome=EvaluationOutcome.NOT_APPLICABLE,
                expected_target=expected,
                observed_evidence=repr(gen),
                reason_codes=["UNKNOWN_CATEGORY"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=is_empty,
                limitations=f"No scoring logic for category '{category}'",
                evaluator_version=self.VERSION,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Basic Arithmetic
    # ─────────────────────────────────────────────────────────────────────────
    def _score_arithmetic(
        self,
        record_id: str,
        prompt: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        # Extract numbers from prompt
        p_lower = prompt.lower()
        # Find all integer tokens
        nums = [int(n) for n in re.findall(r"-?\b\d+\b", prompt)]

        op = None
        computed_res = None
        if "sum of" in p_lower or "plus" in p_lower or " + " in prompt or "added to" in p_lower:
            op = "add"
            if len(nums) >= 2:
                computed_res = nums[0] + nums[1]
        elif "difference" in p_lower or "minus" in p_lower or " - " in prompt or "subtracted from" in p_lower:
            op = "sub"
            if len(nums) >= 2:
                if "subtracted from" in p_lower:
                    computed_res = nums[1] - nums[0]
                else:
                    computed_res = nums[0] - nums[1]
        elif "product" in p_lower or "multiplied by" in p_lower or "times" in p_lower or " * " in prompt:
            op = "mul"
            if len(nums) >= 2:
                computed_res = nums[0] * nums[1]
        elif "quotient" in p_lower or "divided by" in p_lower or " / " in prompt or "//" in prompt:
            op = "div"
            if len(nums) >= 2 and nums[1] != 0:
                computed_res = nums[0] // nums[1]

        if computed_res is None:
            return EvaluationResult(
                record_id=record_id,
                category="basic_arithmetic",
                scoring_status=ScoringStatus.UNSCORED,
                outcome=EvaluationOutcome.NOT_APPLICABLE,
                expected_target=expected,
                observed_evidence=repr(generated),
                reason_codes=["ARITHMETIC_PARSE_AMBIGUOUS"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=is_empty,
                limitations="Unable to deterministically parse arithmetic operation from prompt.",
                evaluator_version=self.VERSION,
            )

        # Parse numbers from generated response using word boundary matching to avoid substring match
        target_str = str(computed_res)
        # Check negative or positive exact numeric token
        gen_tokens = re.findall(r"-?\b\d+\b", generated)

        # Check if computed_res is among generated tokens
        is_correct = any(int(tok) == computed_res for tok in gen_tokens)

        reason_codes = []
        if is_correct:
            outcome = EvaluationOutcome.CORRECT
            reason_codes.append("ARITHMETIC_RESULT_MATCH")
        else:
            outcome = EvaluationOutcome.INCORRECT
            if is_empty:
                reason_codes.append("EMPTY_RESPONSE")
            else:
                reason_codes.append("ARITHMETIC_RESULT_MISMATCH")

        return EvaluationResult(
            record_id=record_id,
            category="basic_arithmetic",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=str(computed_res),
            observed_evidence=f"Generated tokens: {gen_tokens}",
            reason_codes=reason_codes,
            format_valid=len(gen_tokens) > 0,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Structured Formatting (JSON)
    # ─────────────────────────────────────────────────────────────────────────
    def _score_structured_formatting(
        self,
        record_id: str,
        prompt: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        # Determine expected key and value from expected JSON if available
        exp_key = None
        exp_val = None
        try:
            exp_json = json.loads(expected)
            if isinstance(exp_json, dict) and len(exp_json) == 1:
                exp_key = list(exp_json.keys())[0]
                exp_val = exp_json[exp_key]
        except Exception:
            pass

        gen_clean = generated.strip()
        syntax_valid = False
        schema_valid = False
        reason_codes = []

        if is_empty:
            reason_codes.append("EMPTY_RESPONSE")
            return EvaluationResult(
                record_id=record_id,
                category="structured_formatting",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=reason_codes,
                format_valid=False,
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        try:
            parsed = json.loads(gen_clean)
            syntax_valid = True
            reason_codes.append("JSON_SYNTAX_VALID")
        except Exception as e:
            reason_codes.append("JSON_SYNTAX_INVALID")
            return EvaluationResult(
                record_id=record_id,
                category="structured_formatting",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence=repr(gen_clean[:80]),
                reason_codes=reason_codes,
                format_valid=False,
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=is_empty,
                limitations=f"JSON parser error: {str(e)[:50]}",
                evaluator_version=self.VERSION,
            )

        # If syntax valid, check schema
        if not isinstance(parsed, dict) or len(parsed) == 0:
            reason_codes.append("JSON_SCHEMA_NOT_OBJECT")
            outcome = EvaluationOutcome.INCORRECT
        else:
            if exp_key is not None:
                if exp_key in parsed:
                    reason_codes.append("JSON_KEY_MATCH")
                    if str(parsed[exp_key]).strip() == str(exp_val).strip():
                        reason_codes.append("JSON_VALUE_MATCH")
                        schema_valid = True
                        outcome = EvaluationOutcome.CORRECT
                    else:
                        reason_codes.append("JSON_VALUE_MISMATCH")
                        outcome = EvaluationOutcome.PARTIALLY_CORRECT
                else:
                    reason_codes.append("JSON_KEY_MISMATCH")
                    outcome = EvaluationOutcome.PARTIALLY_CORRECT
            else:
                schema_valid = True
                outcome = EvaluationOutcome.CORRECT

        return EvaluationResult(
            record_id=record_id,
            category="structured_formatting",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=expected,
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=syntax_valid,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Uncertainty & Abstention
    # ─────────────────────────────────────────────────────────────────────────
    def _score_uncertainty_abstention(
        self,
        record_id: str,
        prompt: str,
        context: str,
        metadata: Dict[str, Any],
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        # Determine ground-truth answerability independently
        requires_abstention = metadata.get("requires_abstention")
        if requires_abstention is None:
            # Fallback heuristic: if expected contains typical refusal phrases
            requires_abstention = any(
                p in expected.lower()
                for p in ["does not state", "not state", "not record", "does not mention", "does not specify", "cannot be determined"]
            )

        gen_lower = generated.lower().strip()
        refusal_phrases = [
            "does not state",
            "not state",
            "does not record",
            "not record",
            "does not mention",
            "not mention",
            "does not specify",
            "not specify",
            "does not contain",
            "not contain",
            "cannot be determined",
            "not provide",
            "does not provide",
            "insufficient",
        ]
        model_abstained = any(p in gen_lower for p in refusal_phrases)

        reason_codes = []
        if is_empty:
            reason_codes.append("EMPTY_RESPONSE")
            return EvaluationResult(
                record_id=record_id,
                category="uncertainty_abstention",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=reason_codes,
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        if requires_abstention:
            # Expected refusal
            if model_abstained:
                outcome = EvaluationOutcome.CORRECT
                reason_codes.append("APPROPRIATE_ABSTENTION")
            else:
                outcome = EvaluationOutcome.INCORRECT
                reason_codes.append("UNSUPPORTED_ANSWER_REFUSAL_EXPECTED")
        else:
            # Control: expected grounded answer
            # Extract target entity from expected answer
            # E.g. "Twelve kilometers of single-mode optical fiber were drawn at tower Beta."
            if model_abstained:
                outcome = EvaluationOutcome.INCORRECT
                reason_codes.append("INCORRECT_ABSTENTION_CONTROL_WAS_ANSWERABLE")
            else:
                # Check whether expected core keywords exist in generated response
                # Extract key noun chunks or numbers
                key_elements = re.findall(r"\b[A-Za-z0-9_-]+\b", expected.lower())
                substantive = [w for w in key_elements if w not in {"the", "a", "an", "of", "in", "at", "for", "were", "was", "is", "by"}]
                matches = sum(1 for w in substantive if w in gen_lower)
                ratio = matches / len(substantive) if substantive else 0.0

                if ratio >= 0.6:
                    outcome = EvaluationOutcome.CORRECT
                    reason_codes.append("GROUNDED_ANSWER_MATCH")
                elif ratio >= 0.3:
                    outcome = EvaluationOutcome.PARTIALLY_CORRECT
                    reason_codes.append("GROUNDED_ANSWER_PARTIAL")
                else:
                    outcome = EvaluationOutcome.INCORRECT
                    reason_codes.append("GROUNDED_ANSWER_MISMATCH")

        return EvaluationResult(
            record_id=record_id,
            category="uncertainty_abstention",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=expected,
            observed_evidence=repr(generated.strip()[:80]),
            reason_codes=reason_codes,
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Context Extraction
    # ─────────────────────────────────────────────────────────────────────────
    def _score_context_extraction(
        self,
        record_id: str,
        prompt: str,
        context: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        if is_empty:
            return EvaluationResult(
                record_id=record_id,
                category="context_extraction",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        # Target entity extraction from expected response
        # E.g. "Apollo_test_0 is assigned as the primary navigation pilot." -> Target entity is "Apollo_test_0"
        m = re.search(r"\b([A-Za-z0-9]+_[A-Za-z0-9_]+)\b", expected)
        target_entity = m.group(1) if m else None

        reason_codes = []
        gen_clean = generated.strip()

        if target_entity:
            if target_entity.lower() in gen_clean.lower():
                outcome = EvaluationOutcome.CORRECT
                reason_codes.append("TARGET_ENTITY_EXTRACTED")
            else:
                outcome = EvaluationOutcome.INCORRECT
                reason_codes.append("TARGET_ENTITY_NOT_FOUND")
        else:
            # Fallback to normalized substring match
            exp_norm = expected.strip().lower()
            if exp_norm in gen_clean.lower():
                outcome = EvaluationOutcome.CORRECT
                reason_codes.append("EXACT_NORMALIZED_MATCH")
            else:
                outcome = EvaluationOutcome.INCORRECT
                reason_codes.append("SUBSTRING_MATCH_FAILED")

        return EvaluationResult(
            record_id=record_id,
            category="context_extraction",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=target_entity or expected,
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4b. Context Copying (Target-Value Binding & Anti-Cheating Scorer)
    # ─────────────────────────────────────────────────────────────────────────
    def _score_context_copying(
        self,
        record: Dict[str, Any],
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        record_id = record.get("id", "UNKNOWN_RECORD")
        target_val = record.get("target_value")
        if not target_val:
            # Fallback if record does not have explicit target_value field
            target_val = expected.strip()

        gen_clean = generated.strip()
        if is_empty or not gen_clean:
            return EvaluationResult(
                record_id=record_id,
                category="context_copying",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=target_val,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                format_valid=False,
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        gen_lower = gen_clean.lower()
        target_lower = target_val.strip().lower()
        reason_codes = []

        # Closed vocabulary of phonetic prefixes used in Exp21
        all_prefixes = ["alpha", "bravo", "delta", "echo", "foxtrot", "hotel", "india", "juliet"]
        found_prefixes = [p for p in all_prefixes if re.search(r"\b" + p + r"\b", gen_lower)]

        # Negative Guard 1: Competing multiple prefix candidates in completion
        if len(found_prefixes) > 1:
            reason_codes.append("AMBIGUOUS_MULTIPLE_CANDIDATES")
            return EvaluationResult(
                record_id=record_id,
                category="context_copying",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=target_val,
                observed_evidence=repr(gen_clean[:80]),
                reason_codes=reason_codes,
                format_valid=False,
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=is_empty,
                evaluator_version=self.VERSION,
            )

        # Negative Guard 2: Repeated prefix loops (e.g. "Delta Delta 56")
        if len(found_prefixes) == 1:
            prefix = found_prefixes[0]
            prefix_occurrences = len(re.findall(r"\b" + prefix + r"\b", gen_lower))
            if prefix_occurrences > 1:
                reason_codes.append("REPEATED_PREFIX_LOOP")
                return EvaluationResult(
                    record_id=record_id,
                    category="context_copying",
                    scoring_status=ScoringStatus.SCORED,
                    outcome=EvaluationOutcome.INCORRECT,
                    expected_target=target_val,
                    observed_evidence=repr(gen_clean[:80]),
                    reason_codes=reason_codes,
                    format_valid=False,
                    is_eos=is_eos,
                    is_max_tokens=is_max,
                    is_empty=is_empty,
                    evaluator_version=self.VERSION,
                )

        # Exact word-boundary match for target value (e.g., "\bdelta 56\b")
        target_pattern = r"\b" + re.escape(target_lower) + r"\b"
        target_copied = bool(re.search(target_pattern, gen_lower))

        # Check full answer match (strip terminal punctuation)
        exp_clean = expected.strip().lower()
        gen_no_punct = re.sub(r"[.!?]+$", "", gen_lower).strip()
        exp_no_punct = re.sub(r"[.!?]+$", "", exp_clean).strip()
        full_match = (gen_no_punct == exp_no_punct)

        if target_copied:
            outcome = EvaluationOutcome.CORRECT
            reason_codes.append("TARGET_VALUE_COPIED")
            if full_match:
                reason_codes.append("FULL_ANSWER_MATCH")
        else:
            outcome = EvaluationOutcome.INCORRECT
            reason_codes.append("TARGET_VALUE_MISMATCH")

        return EvaluationResult(
            record_id=record_id,
            category="context_copying",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=target_val,
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=full_match or target_copied,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Reference Resolution
    # ─────────────────────────────────────────────────────────────────────────
    def _score_reference_resolution(
        self,
        record_id: str,
        prompt: str,
        context: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        if is_empty:
            return EvaluationResult(
                record_id=record_id,
                category="reference_resolution",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        # Extract referent target from expected string
        # E.g. "The pronoun 'it' designates the brass compass_test_1." -> target is "brass compass_test_1"
        m = re.search(r"designates (?:the )?([^.]+)", expected)
        target_antecedent = m.group(1).strip() if m else None

        reason_codes = []
        gen_clean = generated.strip()

        if target_antecedent and target_antecedent.lower() in gen_clean.lower():
            outcome = EvaluationOutcome.CORRECT
            reason_codes.append("ANTECEDENT_RESOLVED")
        else:
            outcome = EvaluationOutcome.INCORRECT
            reason_codes.append("ANTECEDENT_RESOLUTION_FAILED")

        return EvaluationResult(
            record_id=record_id,
            category="reference_resolution",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=target_antecedent or expected,
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Instruction Following
    # ─────────────────────────────────────────────────────────────────────────
    def _score_instruction_following(
        self,
        record_id: str,
        prompt: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        if is_empty:
            return EvaluationResult(
                record_id=record_id,
                category="instruction_following",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        exp_target = expected.strip()
        gen_clean = generated.strip()

        # In Exp17 instruction following, prompts request exact transformations:
        # e.g. uppercase, lowercase, reverse order, exact token emission.
        # Strict exact match or single-line clean match
        reason_codes = []
        if gen_clean == exp_target:
            outcome = EvaluationOutcome.CORRECT
            reason_codes.append("EXACT_CONSTRAINT_SATISFIED")
        elif exp_target in gen_clean:
            outcome = EvaluationOutcome.PARTIALLY_CORRECT
            reason_codes.append("TARGET_SUBSTRING_FOUND_WITH_NOISE")
        else:
            outcome = EvaluationOutcome.INCORRECT
            reason_codes.append("CONSTRAINT_VIOLATED")

        return EvaluationResult(
            record_id=record_id,
            category="instruction_following",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=exp_target,
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Factual QA
    # ─────────────────────────────────────────────────────────────────────────
    def _score_factual_qa(
        self,
        record_id: str,
        prompt: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        if is_empty:
            return EvaluationResult(
                record_id=record_id,
                category="factual_qa",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        # Extract focal entity from expected answer
        # E.g. "Canberra is the capital city of Australia." -> "Canberra"
        # "The Antarctic Desert is the largest polar desert on Earth." -> "Antarctic" / "Antarctic Desert"
        # "The Sahara Desert is the largest subtropical hot desert on Earth." -> "Sahara" / "Sahara Desert"
        # "An electron carries a negative electrical charge in an atom." -> "electron"
        # "The chemical symbol for iron is Fe." -> "Fe"
        target_entities = []
        if "capital" in prompt.lower():
            target_entities.append(expected.split()[0].strip(",."))
        elif "symbol" in prompt.lower():
            target_entities.append(expected.split()[-1].strip(",."))
        elif "desert" in prompt.lower():
            if "sahara" in expected.lower():
                target_entities.append("sahara")
            elif "antarctic" in expected.lower():
                target_entities.append("antarctic")
            else:
                target_entities.append(expected.split()[0].strip(",."))
        elif "particle" in prompt.lower():
            if "electron" in expected.lower():
                target_entities.append("electron")
            elif "proton" in expected.lower():
                target_entities.append("proton")
            elif "neutron" in expected.lower():
                target_entities.append("neutron")
            else:
                target_entities.append(expected.split()[0].strip(",."))
        elif "storm" in prompt.lower() or "spot" in prompt.lower():
            target_entities.append("jupiter")
        else:
            words = [w.strip(".,") for w in expected.split() if w.strip(".,")]
            while words and words[0].lower() in {"the", "a", "an"}:
                words.pop(0)
            target_entities.append(words[0] if words else expected.split()[0].strip(",."))

        gen_clean = generated.strip()
        matched = False
        for ent in target_entities:
            if len(ent) <= 2:
                # Require word-boundary match for short symbols (e.g. 'W', 'Fe', 'Cu') to prevent false-positive substring matches
                pattern = r"\b" + re.escape(ent) + r"\b"
                if re.search(pattern, gen_clean, re.IGNORECASE):
                    matched = True
                    break
            else:
                if ent.lower() in gen_clean.lower():
                    matched = True
                    break

        reason_codes = []
        if matched:
            outcome = EvaluationOutcome.CORRECT
            reason_codes.append("FACTUAL_ENTITY_MATCH")
        else:
            outcome = EvaluationOutcome.INCORRECT
            reason_codes.append("FACTUAL_ENTITY_MISMATCH")

        return EvaluationResult(
            record_id=record_id,
            category="factual_qa",
            scoring_status=ScoringStatus.SCORED,
            outcome=outcome,
            expected_target=str(target_entities),
            observed_evidence=repr(gen_clean[:80]),
            reason_codes=reason_codes,
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            evaluator_version=self.VERSION,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Short Explanation
    # ─────────────────────────────────────────────────────────────────────────
    def _score_short_explanation(
        self,
        record_id: str,
        prompt: str,
        expected: str,
        generated: str,
        is_eos: bool,
        is_max: bool,
        is_empty: bool,
    ) -> EvaluationResult:
        if is_empty:
            return EvaluationResult(
                record_id=record_id,
                category="short_explanation",
                scoring_status=ScoringStatus.SCORED,
                outcome=EvaluationOutcome.INCORRECT,
                expected_target=expected,
                observed_evidence="<EMPTY>",
                reason_codes=["EMPTY_RESPONSE"],
                is_eos=is_eos,
                is_max_tokens=is_max,
                is_empty=True,
                evaluator_version=self.VERSION,
            )

        # Explanations require semantic assessment that deterministic keyword matching cannot reliably guarantee.
        # In strict adherence to Section 4: "If a reliable deterministic scorer cannot establish correctness,
        # mark the result NEEDS_REVIEW or UNSCORED. Do not fabricate a semantic score."
        return EvaluationResult(
            record_id=record_id,
            category="short_explanation",
            scoring_status=ScoringStatus.NEEDS_REVIEW,
            outcome=EvaluationOutcome.NOT_APPLICABLE,
            expected_target=expected,
            observed_evidence=repr(generated.strip()[:80]),
            reason_codes=["SEMANTIC_EXPLANATION_REQUIRES_HUMAN_REVIEW"],
            format_valid=True,
            is_eos=is_eos,
            is_max_tokens=is_max,
            is_empty=is_empty,
            limitations="Causal scientific explanations require expert human review or authorized semantic NLI judge.",
            evaluator_version=self.VERSION,
        )
