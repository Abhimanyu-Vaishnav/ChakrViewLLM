"""
ChakrView v0.5.0 Release Verification Script.

Executes a strict 14-point verification audit prior to founder release approval:
1. Working tree cleanliness (or release files only)
2. Expected commit ancestry (1e06746)
3. Canonical parameter count (3,443,136)
4. Canonical weight SHA-256 (c5571c...00a282da)
5. Baseline Delta W = 0
6. Historical v0.2.0 verification
7. V0.4 candidate regression
8. V0.5 candidate regression
9. V0.6 candidate regression
10. Secret and credential leak audit
11. Local machine path audit in release documents
12. Release manifest presence (V0.5.0_RELEASE_MANIFEST.md)
13. Founder test documentation presence (FOUNDER_TEST_RESULTS.md)
14. Required project documentation presence (README, LICENSE, CHANGELOG)

Usage:
    python scripts/verify_v050_release.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe Windows stream reconfigure
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import (
    verify_canonical_baseline,
    CANONICAL_BASELINE_SHA256,
    CANONICAL_PARAM_COUNT,
)


def print_check(index: int, name: str, passed: bool, detail: str = ""):
    dots = "." * (45 - len(name))
    status = "PASS" if passed else "FAIL"
    msg = f"[{index:02d}] {name} {dots} {status}"
    if detail:
        msg += f" ({detail})"
    print(msg)


def verify_all() -> bool:
    print("=" * 70)
    print("CHAKRVIEW v0.5.0 RELEASE CANDIDATE VERIFICATION")
    print("=" * 70)
    print(f"Target Release: v0.5.0 | Candidate Ancestry Target: 1e06746\n")

    checks: List[Tuple[int, str, bool, str]] = []

    # 1. Clean working tree check
    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        porcelain = res.stdout.strip()
        allowed_release_prefixes = (
            "scripts/founder_manual_test.py",
            "scripts/verify_v050_release.py",
            "chakrview/runtime/hardware_adaptive_engine.py",
            "chakrview/cognition/language_realizer.py",
            "chakrview/cognition/language_understanding.py",
            "chakrview/cognition/strict_recurrent_i4_evaluation.py",
            "tests/test_hardware_adaptive_engine.py",
            "tests/test_language_realizer.py",
            "tests/test_language_understanding.py",
            "docs/founder_tests/",
            "docs/releases/",
            "README.md",
        )
        untracked = []
        for line in porcelain.splitlines():
            # git status --porcelain has 2 characters for status (e.g. ' M' or '??') then a space
            path = line[2:].strip()
            if not any(path.startswith(prefix) for prefix in allowed_release_prefixes):
                untracked.append(path)
        tree_clean = len(untracked) == 0
        checks.append((1, "Clean Working Tree & Governed Release Files", tree_clean, "All release files governed" if tree_clean else f"extraneous: {untracked}"))
    except Exception as e:
        checks.append((1, "Clean Working Tree & Governed Release Files", False, str(e)))

    # 2. Expected commit ancestry
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        head_commit = res.stdout.strip()
        # Check if 1e06746 is ancestor of HEAD or HEAD itself
        res2 = subprocess.run(["git", "merge-base", "--is-ancestor", "1e06746", "HEAD"], cwd=REPO_ROOT)
        ancestor_ok = (res2.returncode == 0)
        checks.append((2, "Expected Commit Ancestry (1e06746)", ancestor_ok, f"HEAD: {head_commit[:7]}"))
    except Exception as e:
        checks.append((2, "Expected Commit Ancestry (1e06746)", False, str(e)))

    # 3. Canonical parameter count
    try:
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        total_p = sum(p.numel() for p in model.parameters())
        p_ok = (total_p == CANONICAL_PARAM_COUNT)
        checks.append((3, "Canonical Parameter Count (3,443,136)", p_ok, f"{total_p:,} params"))
    except Exception as e:
        checks.append((3, "Canonical Parameter Count (3,443,136)", False, str(e)))

    # 4. Canonical weight SHA-256
    try:
        rep = verify_canonical_baseline(model)
        sha_ok = rep["is_bit_exact"] and (rep["digest"] == CANONICAL_BASELINE_SHA256)
        checks.append((4, "Canonical Weight SHA-256 Bit-Exactness", sha_ok, f"{rep['digest'][:16]}..."))
    except Exception as e:
        checks.append((4, "Canonical Weight SHA-256 Bit-Exactness", False, str(e)))

    # 5. Baseline Delta W = 0
    delta_w_ok = (rep.get("delta_w", -1) == 0) if "rep" in locals() else False
    checks.append((5, "Baseline Weight Drift (Delta W = 0)", delta_w_ok, f"Delta W = {rep.get('delta_w', -1)}"))

    # 6. Historical v0.2.0 verification
    v020_path = REPO_ROOT / "scripts" / "verify_v020.py"
    checks.append((6, "Historical v0.2.0 Verification Script Present", v020_path.exists(), str(v020_path.name)))

    # 7. V0.4 Candidate Regression
    try:
        from chakrview.cognition.v040_candidate.v040_master_gate import V040CandidateMasterGate
        v4_rep = V040CandidateMasterGate.evaluate_candidate()
        v4_ok = (v4_rep.audit_verdict == "V040_RELEASE_CANDIDATE_READY" and v4_rep.all_gates_pass)
        checks.append((7, "V0.4 Cognitive Architecture Regression", v4_ok, v4_rep.audit_verdict))
    except Exception as e:
        checks.append((7, "V0.4 Cognitive Architecture Regression", False, str(e)))

    # 8. V0.5 Candidate Regression
    try:
        from chakrview.cognition.v050_candidate.v050_master_gate import V050CandidateMasterGate
        v5_rep = V050CandidateMasterGate.evaluate_candidate()
        v5_ok = (v5_rep.final_status == "V050_NEURAL_CAPABILITY_PARTIAL" and v5_rep.canonical_baseline_intact)
        checks.append((8, "V0.5 Neural Capability Regression", v5_ok, v5_rep.final_status))
    except Exception as e:
        checks.append((8, "V0.5 Neural Capability Regression", False, str(e)))

    # 9. V0.6 Candidate Regression
    try:
        from chakrview.cognition.v060_candidate.v060_master_gate import V060CandidateMasterGate
        v6_rep = V060CandidateMasterGate.evaluate_candidate()
        v6_ok = (v6_rep.final_verdict == "V060_NEURAL_SEMANTIC_CAPABILITY_PARTIAL" and v6_rep.canonical_baseline_intact)
        checks.append((9, "V0.6 Neural Semantic Regression", v6_ok, v6_rep.final_verdict))
    except Exception as e:
        checks.append((9, "V0.6 Neural Semantic Regression", False, str(e)))

    # 10. Secret and Credential Leak Audit
    suspect_patterns = [
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
        re.compile(r"(?i)ghp_[A-Za-z0-9]{36}"),
        re.compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key"),
    ]
    secret_leaks = 0
    docs_to_scan = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "releases" / "V0.5.0_RELEASE_MANIFEST.md",
        REPO_ROOT / "docs" / "founder_tests" / "v0.5.0" / "FOUNDER_TEST_RESULTS.md",
    ]
    for dpath in docs_to_scan:
        if dpath.exists():
            content = dpath.read_text(encoding="utf-8")
            for pat in suspect_patterns:
                if pat.search(content):
                    secret_leaks += 1
    checks.append((10, "Secret & Credential Leaks Audit", secret_leaks == 0, f"{secret_leaks} patterns detected"))

    # 11. Accidental Local Machine Paths Audit in release docs
    local_path_pattern = re.compile(r"(?:[a-zA-Z]:\\Users\\[a-zA-Z0-9_\-]+|/Users/[a-zA-Z0-9_\-]+|/home/[a-zA-Z0-9_\-]+)")
    path_leaks = 0
    for dpath in docs_to_scan:
        if dpath.exists():
            content = dpath.read_text(encoding="utf-8")
            if local_path_pattern.search(content):
                path_leaks += 1
    checks.append((11, "No Local Machine Paths in Release Docs", path_leaks == 0, f"{path_leaks} files with local paths"))

    # 12. Release Manifest Exists
    manifest_path = REPO_ROOT / "docs" / "releases" / "V0.5.0_RELEASE_MANIFEST.md"
    checks.append((12, "Release Manifest Exists (v0.5.0)", manifest_path.exists(), str(manifest_path.name)))

    # 13. Founder Test Documentation Exists
    ft_path = REPO_ROOT / "docs" / "founder_tests" / "v0.5.0" / "FOUNDER_TEST_RESULTS.md"
    checks.append((13, "Founder Test Ledger Exists", ft_path.exists(), str(ft_path.name)))

    # 14. Required Project Documentation
    req_docs = [REPO_ROOT / "README.md", REPO_ROOT / "LICENSE", REPO_ROOT / "CHANGELOG.md"]
    all_req_docs = all(p.exists() for p in req_docs)
    checks.append((14, "Required Core Docs (README, LICENSE, CHANGELOG)", all_req_docs, "Present"))

    # Render check results
    all_passed = True
    for idx, name, passed, detail in checks:
        print_check(idx, name, passed, detail)
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("VERIFICATION VERDICT: PASS (Candidate is ready for Founder testing)")
        print("NEXT STEP: Founder manual testing via scripts/founder_manual_test.py")
        print("=" * 70)
        return True
    else:
        print("VERIFICATION VERDICT: FAIL (Release blockers detected)")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
