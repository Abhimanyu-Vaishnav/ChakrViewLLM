"""
Strict Evaluators R2 Package for UCIP-10E-R2.

Mandatory Enhancements:
1. Fix cyclic keyword-stuffed repetition vulnerability:
   - Detects repeated n-gram phrase cycles (e.g. "ram storage memory ram storage memory...")
   - Rejects cyclic word-salad loops while allowing legitimate semantic repetition.
2. Distinct evaluation of Devanagari script vs Romanized Hindi.
3. Strict arithmetic validation with RHS/word problem answer matching.
4. Separate verbatim vs normalized copying metrics.
5. Deductive logic conclusion verification.
6. Epistemic abstention vs false confidence classification.
7. Runtime-state awareness vs static identity recall.
"""

from __future__ import annotations

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

def detect_phrase_repetition_or_cycles(text: str) -> Tuple[bool, str]:
    """
    Detects consecutive word repetitions and cyclic phrase loops.
    E.g. catches 'ram storage memory ram storage memory ram storage memory'
    as well as 'cpu cpu cpu'.
    """
    words = [w.strip().lower() for w in text.split() if w.strip()]
    n = len(words)
    if n < 3:
        return False, "ok"

    # 1. Consecutive identical word loop (3 in a row)
    for i in range(n - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True, f"consecutive_word_loop: '{words[i]}'"

    # 2. Cyclic phrase repetition: check pattern lengths k from 2 to 6
    for k in range(2, 7):
        if n >= 2 * k:
            for i in range(n - 2 * k + 1):
                phrase1 = words[i : i + k]
                phrase2 = words[i + k : i + 2 * k]
                if phrase1 == phrase2:
                    # If this phrase repeats 2 or more times back-to-back, check if it dominates
                    if n >= i + 3 * k:
                        phrase3 = words[i + 2 * k : i + 3 * k]
                        if phrase1 == phrase3:
                            return True, f"cyclic_phrase_loop: {' '.join(phrase1)}"
                    # For k >= 3, even 2 full repetitions of a multi-word sequence is degenerate in short completions
                    if k >= 3 and n <= 2 * k + 4:
                        return True, f"cyclic_phrase_loop: {' '.join(phrase1)}"

    return False, "ok"


class StrictEvaluatorsR2:

    @staticmethod
    def evaluate_language_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """
        Repairs R1 evaluator by incorporating cyclic phrase loop detection
        alongside semantic rubric, script, and keyword checks.
        """
        text = raw_output.strip()
        norm = text.lower()
        words = text.split()

        if not text:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OUTPUT"}

        has_cycle, cycle_reason = detect_phrase_repetition_or_cycles(text)
        if has_cycle:
            return {"passed": False, "score": 0.0, "reason": f"DEGENERATE_REPETITION ({cycle_reason})"}

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
    def evaluate_arithmetic_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        """Strict deterministic arithmetic evaluator."""
        text = raw_output.strip()
        if not text:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OUTPUT", "extracted": None}
        has_cycle, cycle_reason = detect_phrase_repetition_or_cycles(text)
        if has_cycle:
            return {"passed": False, "score": 0.0, "reason": f"DEGENERATE_REPETITION ({cycle_reason})", "extracted": None}

        tgt = probe["target_number"]
        if tgt is None:
            return {"passed": False, "score": 0.0, "reason": "NO_TARGET_DEFINED", "extracted": None}

        # Equation format: RHS check
        if "=" in text:
            lhs, rhs = text.rsplit("=", 1)
            rhs_nums = [float(m) if "." in m else int(m) for m in re.findall(r"-?\b\d+(?:\.\d+)?\b", rhs)]
            if rhs_nums:
                ans = rhs_nums[0]
                passed = (abs(ans - tgt) < 1e-5)
                return {
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "reason": f"Equation RHS is {ans}, expected {tgt}",
                    "extracted": ans
                }

        # Word problem phrasing: e.g. "have 3 remaining"
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

        # Solitary number
        clean_solitary = re.sub(r"[^\d\.-]", "", text)
        if clean_solitary:
            try:
                val = float(clean_solitary) if "." in clean_solitary else int(clean_solitary)
                if abs(val - tgt) < 1e-5:
                    return {"passed": True, "score": 1.0, "reason": f"Exact numeric match {val}", "extracted": val}
            except ValueError:
                pass

        all_nums = [float(m) if "." in m else int(m) for m in re.findall(r"-?\b\d+(?:\.\d+)?\b", text)]
        if tgt in all_nums and len(all_nums) == 1:
            return {"passed": True, "score": 1.0, "reason": f"Single target number {tgt} present", "extracted": tgt}

        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Target {tgt} not asserted. Found: {all_nums}",
            "extracted": all_nums[-1] if all_nums else None
        }

    @staticmethod
    def evaluate_copying_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()
        tgt = probe.get("target_text", "").strip()

        if not text:
            return {"passed": False, "score": 0.0, "exact_match": False, "norm_match": False, "reason": "EMPTY_OUTPUT"}
        has_cycle, cycle_reason = detect_phrase_repetition_or_cycles(text)
        if has_cycle:
            return {"passed": False, "score": 0.0, "exact_match": False, "norm_match": False, "reason": f"REPETITIVE ({cycle_reason})"}

        exact_match = (text == tgt)
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
            "reason": f"Exact: {exact_match}, Norm: {norm_match} ('{tgt}' vs '{text}')"
        }

    @staticmethod
    def evaluate_logic_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()
        norm = text.lower()
        has_cycle, _ = detect_phrase_repetition_or_cycles(text)
        if not text or has_cycle:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        exp_ans = str(probe.get("expected_answer", "")).lower().strip()
        passed = (re.search(rf"\b{re.escape(exp_ans)}\b", norm) is not None)
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Asserted '{exp_ans}': {passed}"}

    @staticmethod
    def evaluate_uncertainty_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip().lower()
        has_cycle, _ = detect_phrase_repetition_or_cycles(text)
        if not text or has_cycle:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        abstention_markers = [
            "cannot answer", "not enough information", "cannot determine",
            "insufficient", "unknown", "impossible to determine", "missing", "unclear",
            "not specified"
        ]
        has_abstention = any(m in text for m in abstention_markers)
        should_abstain = probe.get("metadata", {}).get("should_abstain", True)
        passed = (has_abstention == should_abstain)
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Abstention match: {passed}"}

    @staticmethod
    def evaluate_coding_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
        code = code_match.group(1) if code_match else text

        test_code = probe.get("code_test", {}).get("test_code", "")
        if not test_code:
            return {"passed": False, "score": 0.0, "reason": "NO_TEST_CODE"}

        full_code = f"{code}\n\n{test_code}"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            res = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=5)
            passed = (res.returncode == 0)
            return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": "Execution passed" if passed else f"Error: {res.stderr[:80]}"}
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
    def evaluate_runtime_awareness_r2(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
        norm = raw_output.strip().lower()
        has_cycle, _ = detect_phrase_repetition_or_cycles(norm)
        if not norm or has_cycle:
            return {"passed": False, "score": 0.0, "reason": "EMPTY_OR_DEGENERATE"}

        exp_kws = probe.get("expected_keywords", [])
        hits = [k for k in exp_kws if k.lower() in norm]
        passed = len(hits) >= 1
        return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": f"Hits: {hits}"}
