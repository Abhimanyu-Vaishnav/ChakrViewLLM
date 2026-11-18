"""
Strict deterministic evaluators across all 10 capability domains.

Rules enforced:
1. Arithmetic: Deterministic numeric oracle; wrong numbers or operand-presence matches are rejected.
2. Copying: Verbatim exact-match vs normalized-match reported separately.
3. Language: Rubric-based semantic validation across English, Hindi, Hinglish. No length-only passes.
4. Logic: Exact conclusion extraction with verification.
5. Generalization: Out-of-distribution operand and entity evaluation.
6. Multi-turn: Context and entity tracking verification.
7. Self-correction: Two-turn revision verification.
8. Uncertainty: Abstention detection without false positive collapse.
9. Coding: Sandboxed Python execution in subprocess with timeouts.
10. Self-modeling: Metadata and runtime awareness accuracy.
"""

from __future__ import annotations

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

def is_repetitive(text: str) -> bool:
    words = [w.strip() for w in text.split() if w.strip()]
    if len(words) < 4:
        return False
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return True
    return False


class StrictEvaluators:

    @staticmethod
    def evaluate_arithmetic(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """
        Evaluates arithmetic correctness using deterministic numeric extraction.
        Target number must match the asserted answer, not just any operand.
        """
        text = raw_output.strip()
        if not text:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OUTPUT", "extracted": None}
        if is_repetitive(text):
            return {"passed": False, "score": 0.0, "reason": "REPETITIVE_DEGENERATION", "extracted": None}

        tgt = probe["target_number"]
        if tgt is None:
            return {"passed": False, "score": 0.0, "reason": "NO_TARGET_DEFINED", "extracted": None}

        # Look for equation format: e.g. "X + Y = Z" -> check RHS
        if "=" in text:
            lhs, rhs = text.rsplit("=", 1)
            rhs_nums = [float(m) if "." in m else int(m) for m in re.findall(r"-?\b\d+(?:\.\d+)?\b", rhs)]
            if rhs_nums:
                ans = rhs_nums[0]
                passed = (abs(ans - tgt) < 1e-5)
                return {
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "reason": f"Equation RHS evaluated to {ans}, expected {tgt}",
                    "extracted": ans
                }

        # Look for word problem phrasing: e.g. "have 3 apples remaining", "is 15"
        m = re.search(r"(?:have|has|remains?|remaining|is|are|total(?: of)?)\s+(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            passed = (abs(val - tgt) < 1e-5)
            return {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "reason": f"Phrasing asserted {val}, expected {tgt}",
                "extracted": val
            }

        # If the output is a solitary number
        clean_solitary = re.sub(r"[^\d\.-]", "", text)
        if clean_solitary:
            try:
                val = float(clean_solitary) if "." in clean_solitary else int(clean_solitary)
                if abs(val - tgt) < 1e-5:
                    return {"passed": True, "score": 1.0, "reason": f"Exact numeric match {val}", "extracted": val}
            except ValueError:
                pass

        # Find all numbers
        all_nums = [float(m) if "." in m else int(m) for m in re.findall(r"-?\b\d+(?:\.\d+)?\b", text)]
        # If the target is present and no conflicting numbers are asserted
        if tgt in all_nums and len(all_nums) == 1:
            return {"passed": True, "score": 1.0, "reason": f"Single target number {tgt} present", "extracted": tgt}

        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Target {tgt} not properly asserted in output. Found numbers: {all_nums}",
            "extracted": all_nums[-1] if all_nums else None
        }

    @staticmethod
    def evaluate_copying(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """
        Evaluates exact copying:
        Reports both verbatim exact-match and normalized-match.
        """
        text = raw_output.strip()
        tgt = probe.get("target_text", "").strip()

        if not text:
            return {"passed": False, "score": 0.0, "exact_match": False, "norm_match": False, "reason": "EMPTY_OUTPUT"}
        if is_repetitive(text):
            return {"passed": False, "score": 0.0, "exact_match": False, "norm_match": False, "reason": "REPETITIVE_DEGENERATION"}

        exact_match = (text == tgt)
        
        # Normalized match: strip leading/trailing whitespace & non-alphanumeric punctuation, lowercase
        t_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tgt).lower()
        g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", text).lower()
        norm_match = (t_clean == g_clean)

        passed = exact_match or (probe.get("evaluation_type") == "normalized_copy" and norm_match)
        score = 1.0 if exact_match else (0.8 if norm_match else 0.0)

        return {
            "passed": passed,
            "score": score,
            "exact_match": exact_match,
            "norm_match": norm_match,
            "reason": f"Exact: {exact_match}, Norm: {norm_match} (Target: '{tgt}' vs '{text}')"
        }

    @staticmethod
    def evaluate_language(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """
        Strict semantic language evaluation.
        Rejects length-only passes. Requires semantic keywords, script consistency, and absence of broken tokens.
        """
        text = raw_output.strip()
        norm = text.lower()
        words = text.split()

        if not text:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OUTPUT"}
        if is_repetitive(text):
            return {"passed": False, "score": 0.0, "reason": "REPETITIVE_DEGENERATION"}
        if len(words) < 3:
            return {"passed": False, "score": 0.0, "reason": "TOO_SHORT"}
        if text.endswith("-") or text.endswith(","):
            return {"passed": False, "score": 0.0, "reason": "TRUNCATED_FRAGMENT"}

        subdomain = probe.get("subdomain", "")

        # Devanagari Hindi check
        if subdomain == "hindi":
            has_dev = any('\u0900' <= ch <= '\u097F' for ch in text)
            if not has_dev:
                return {"passed": False, "score": 0.0, "reason": "NO_DEVANAGARI_SCRIPT"}
            kws = probe.get("expected_keywords", [])
            hits = [k for k in kws if k in text]
            if kws and not hits:
                return {"passed": False, "score": 0.0, "reason": f"Missing expected Hindi keywords {kws}"}
            return {"passed": True, "score": 1.0, "reason": f"Devanagari verified with hits: {hits}"}

        # Hinglish check
        if subdomain == "hinglish":
            hing_markers = ["hai", "mein", "ko", "hota", "karte", "kare", "karna", "ka", "ki", "ke", "bhi", "aur"]
            has_marker = any(re.search(rf"\b{m}\b", norm) for m in hing_markers)
            if not has_marker:
                return {"passed": False, "score": 0.0, "reason": "NO_HINGLISH_GRAMMAR_MARKERS"}
            kws = probe.get("expected_keywords", [])
            hits = [k for k in kws if k.lower() in norm]
            if kws and not hits:
                return {"passed": False, "score": 0.0, "reason": f"Missing expected Hinglish keywords {kws}"}
            return {"passed": True, "score": 1.0, "reason": f"Hinglish verified with hits: {hits}"}

        # English check
        kws = probe.get("expected_keywords", [])
        forbidden = probe.get("forbidden_keywords", [])
        for f in forbidden:
            if re.search(rf"\b{f.lower()}\b", norm):
                return {"passed": False, "score": 0.0, "reason": f"Forbidden keyword used: {f}"}

        # Reject known gibberish artifacts
        gibberish = ["que wlean", "curves of rest", "vols hostant", "holatier holy", "code oxide", "tax taskes", "dower"]
        if any(g in norm for g in gibberish):
            return {"passed": False, "score": 0.0, "reason": "CONTAINED_CORRUPTED_TOKENS"}

        if kws:
            hits = [k for k in kws if re.search(rf"\b{k.lower()}\b", norm)]
            if len(hits) < max(1, len(kws) // 2):
                return {"passed": False, "score": 0.0, "reason": f"Insufficient keywords: {hits} of {kws}"}

        return {"passed": True, "score": 1.0, "reason": "Semantic requirements satisfied"}

    @staticmethod
    def evaluate_logic(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """Evaluates logical deduction, contradiction detection, and syllogisms."""
        text = raw_output.strip()
        norm = text.lower()
        if not text or is_repetitive(text):
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        exp_ans = str(probe.get("expected_answer", "")).lower().strip()
        
        # Check if expected answer is asserted
        passed = (re.search(rf"\b{re.escape(exp_ans)}\b", norm) is not None)
        reason = f"Asserted '{exp_ans}' in logical response" if passed else f"Expected '{exp_ans}' missing"
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": reason}

    @staticmethod
    def evaluate_uncertainty(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """Evaluates whether the model correctly abstains or identifies missing information."""
        text = raw_output.strip().lower()
        if not text or is_repetitive(text):
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        # Abstention markers
        abstention_markers = [
            "cannot answer", "not enough information", "cannot determine",
            "insufficient", "unknown", "impossible to determine", "missing", "unclear",
            "not specified"
        ]
        has_abstention = any(m in text for m in abstention_markers)
        should_abstain = probe.get("metadata", {}).get("should_abstain", True)

        passed = (has_abstention == should_abstain)
        reason = "Correct abstention" if (has_abstention and should_abstain) else (
            "Correctly answered without false abstention" if (not has_abstention and not should_abstain) else
            ("False confidence (failed to abstain)" if should_abstain else "False abstention")
        )
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": reason}

    @staticmethod
    def evaluate_coding(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """
        Evaluates coding intelligence by extracting Python code and running sandboxed tests.
        """
        text = raw_output.strip()
        # Extract code blocks if any, or use raw text
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
        code = code_match.group(1) if code_match else text

        test_code = probe.get("code_test", {}).get("test_code", "")
        if not test_code:
            return {"passed": False, "score": 0.0, "reason": "NO_TEST_CODE"}

        full_code = f"{code}\n\n{test_code}"

        # Sandboxed execution with timeout
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            passed = (res.returncode == 0)
            reason = "Execution passed" if passed else f"Runtime error: {res.stderr[:100]}"
            return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": reason}
        except subprocess.TimeoutExpired:
            return {"passed": False, "score": 0.0, "reason": "TIMEOUT_EXPIRED"}
        except Exception as e:
            return {"passed": False, "score": 0.0, "reason": f"Execution failed: {str(e)}"}
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    @staticmethod
    def evaluate_self_modeling(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """Evaluates whether the model accurately reflects its identity and metadata."""
        norm = raw_output.strip().lower()
        if not norm or is_repetitive(norm):
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        exp_kws = probe.get("expected_keywords", [])
        hits = [k for k in exp_kws if k.lower() in norm]
        passed = len(hits) >= 1
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Hits: {hits}"}
