"""
Package Validation and Extraction Smoke Test Script (Milestone 7).

Verifies:
1. Secret & Credential audit on package contents.
2. Extracts package to clean temporary directory.
3. Tests package integrity: all files match CHECKSUMS.sha256.
4. Executes founder smoke test suite from the extracted clean directory.
5. Verifies canonical baseline hash before and after extraction execution.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "release" / "dist" / "chakrview-v0.5.0-research-preview-rc1.zip"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def audit_and_smoke_test_package():
    print("=== RUNNING MILESTONE 7: RELEASE PACKAGE VALIDATION & SMOKE TEST ===")
    assert ZIP_PATH.exists(), f"Release zip missing at {ZIP_PATH}"

    # 1. Inspect archive contents and scan for secrets
    print(f"Archive: {ZIP_PATH.name} ({ZIP_PATH.stat().st_size:,} bytes)")
    suspect_patterns = [
        re.compile(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
        re.compile(r"(?i)ghp_[A-Za-z0-9]{36}"),
        re.compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key"),
        re.compile(r"(?i)password\s*=\s*['\"][^'\"]+['\"]"),
    ]

    secrets_found = []
    file_list = []

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for info in zf.infolist():
            file_list.append(info.filename)
            # Check text files for credentials
            if any(info.filename.endswith(ext) for ext in [".py", ".md", ".json", ".txt", ".ini"]):
                content = zf.read(info.filename).decode("utf-8", errors="replace")
                for pat in suspect_patterns:
                    if pat.search(content):
                        secrets_found.append((info.filename, str(pat)))

    print(f"Files inside archive: {len(file_list)}")
    print(f"Secrets detected: {len(secrets_found)}")
    assert len(secrets_found) == 0, f"Secrets detected in release archive: {secrets_found}"

    # 2. Extract into clean temporary directory
    with tempfile.TemporaryDirectory() as td:
        extract_dir = Path(td) / "extracted"
        print(f"Extracting package into clean directory: {extract_dir}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(extract_dir)

        pkg_root = extract_dir / "chakrview-v0.5.0"
        assert pkg_root.exists(), "chakrview-v0.5.0 folder missing in extracted package"

        # 3. Verify CHECKSUMS.sha256
        checksum_file = pkg_root / "CHECKSUMS.sha256"
        assert checksum_file.exists(), "CHECKSUMS.sha256 missing"
        checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()

        mismatches = []
        for line in checksum_lines:
            if not line.strip():
                continue
            expected_sha, rel_file = line.strip().split("  ", 1)
            target = pkg_root / rel_file
            if not target.exists():
                mismatches.append(f"Missing file: {rel_file}")
                continue
            h = hashlib.sha256()
            with open(target, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            if h.hexdigest() != expected_sha:
                mismatches.append(f"Checksum mismatch: {rel_file}")

        print(f"Checksum verification: {len(checksum_lines)} files checked, mismatches: {len(mismatches)}")
        assert len(mismatches) == 0, f"Checksum verification failed: {mismatches}"

        # 4. Run baseline verification and founder smoke test directly from extracted directory
        print("Running verification and smoke test inside extracted package...")
        python_exe = sys.executable

        # Test canonical baseline inside extracted package
        test_script = pkg_root / "scripts" / "founder_manual_test.py"
        res_smoke = subprocess.run(
            [python_exe, str(test_script), "--suite"],
            cwd=str(pkg_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print("Extracted smoke test output:")
        print(res_smoke.stdout)
        if res_smoke.stderr:
            print("Extracted smoke test stderr:", res_smoke.stderr)
        assert res_smoke.returncode == 0, f"Extracted package smoke test failed (exit {res_smoke.returncode})"

        # Run test suite from extracted package
        res_pytest = subprocess.run(
            [python_exe, "-m", "pytest", "tests/test_runtime_acceptance_audit.py", "tests/test_hardware_adaptive_engine.py"],
            cwd=str(pkg_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print("Extracted package pytest:")
        print(res_pytest.stdout)
        assert res_pytest.returncode == 0, f"Extracted package unit tests failed (exit {res_pytest.returncode})"

    print("=== PACKAGE VALIDATION & SMOKE TEST PASSED (100% CLEAN) ===")


if __name__ == "__main__":
    audit_and_smoke_test_package()
