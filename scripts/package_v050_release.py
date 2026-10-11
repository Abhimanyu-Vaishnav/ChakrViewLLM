"""
Release Packaging Script for ChakrView v0.5.0 Experimental Research Preview.

Creates a clean, reproducible source distribution bundle:
release/chakrview-v0.5.0/
Containing:
- Core library code (chakrview/)
- Tests and verification scripts (tests/, scripts/verify_v050_release.py, scripts/founder_manual_test.py)
- Configuration & Tokenizer vocab/merges (configs/, data/tokenizer_experiments/v4096/)
- Release documentation, manifest, licenses (README.md, CHANGELOG.md, LICENSE, docs/releases/V0.5.0_RELEASE_MANIFEST.md)
- Checksums manifest (CHECKSUMS.sha256)
- Zipped tarball / zip archive

Excludes:
- Large binary training datasets (data/tokenized/, data/raw/)
- Local machine paths, experimental training artifacts, logs, cache directories
"""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "release" / "dist"
BUNDLE_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_release_package() -> Tuple[Path, str, int]:
    print("Building ChakrView v0.5.0 Experimental Research Preview release bundle...")

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Core Source Code (chakrview/)
    shutil.copytree(
        REPO_ROOT / "chakrview",
        BUNDLE_DIR / "chakrview",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )

    # 2. Release & Smoke Test Scripts
    scripts_dest = BUNDLE_DIR / "scripts"
    scripts_dest.mkdir(parents=True, exist_ok=True)
    scripts_to_include = [
        "verify_v050_release.py",
        "verify_v020.py",
        "founder_manual_test.py",
    ]
    for s in scripts_to_include:
        src = REPO_ROOT / "scripts" / s
        if src.exists():
            shutil.copy2(src, scripts_dest / s)

    # 3. Essential Tokenizer & Config Files
    tok_dest = BUNDLE_DIR / "data" / "experiments" / "vocab_4096"
    tok_dest.mkdir(parents=True, exist_ok=True)
    tok_src = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    if tok_src.exists():
        for tf in ["vocab.json", "merges.json", "config.json"]:
            if (tok_src / tf).exists():
                shutil.copy2(tok_src / tf, tok_dest / tf)

    # Configs
    cfg_dest = BUNDLE_DIR / "configs"
    cfg_dest.mkdir(parents=True, exist_ok=True)
    for cf in (REPO_ROOT / "configs").glob("*.json"):
        shutil.copy2(cf, cfg_dest / cf.name)

    # 4. Tests
    tests_dest = BUNDLE_DIR / "tests"
    tests_dest.mkdir(parents=True, exist_ok=True)
    tests_to_include = [
        "test_runtime_acceptance_audit.py",
        "test_hardware_adaptive_engine.py",
        "test_device_placement_inference.py",
        "test_durable_execution_journal.py",
        "test_end_to_end_orchestration_recovery.py",
        "test_language_realizer.py",
        "test_language_understanding.py",
        "test_contract_execution_paths.py",
    ]
    for t in tests_to_include:
        src = REPO_ROOT / "tests" / t
        if src.exists():
            shutil.copy2(src, tests_dest / t)

    # 5. Core Documents & Licenses
    docs_to_copy = ["README.md", "CHANGELOG.md", "LICENSE", "requirements.txt", "pytest.ini"]
    for d in docs_to_copy:
        src = REPO_ROOT / d
        if src.exists():
            shutil.copy2(src, BUNDLE_DIR / d)

    # Release Manifest
    rel_docs_dest = BUNDLE_DIR / "docs" / "releases"
    rel_docs_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REPO_ROOT / "docs" / "releases" / "V0.5.0_RELEASE_MANIFEST.md",
        rel_docs_dest / "V0.5.0_RELEASE_MANIFEST.md",
    )

    # 6. Generate CHECKSUMS.sha256
    checksums_file = BUNDLE_DIR / "CHECKSUMS.sha256"
    checksum_lines = []
    for fpath in sorted(BUNDLE_DIR.rglob("*")):
        if fpath.is_file() and fpath.name != "CHECKSUMS.sha256":
            rel_path = fpath.relative_to(BUNDLE_DIR).as_posix()
            sha = compute_file_sha256(fpath)
            checksum_lines.append(f"{sha}  {rel_path}\n")

    with open(checksums_file, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)

    # 7. Create ZIP archive
    zip_path = DIST_DIR / "chakrview-v0.5.0-research-preview-rc1.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in BUNDLE_DIR.rglob("*"):
            if fpath.is_file():
                arcname = f"chakrview-v0.5.0/{fpath.relative_to(BUNDLE_DIR).as_posix()}"
                zf.write(fpath, arcname)

    archive_sha = compute_file_sha256(zip_path)
    file_count = len(checksum_lines) + 1  # including CHECKSUMS.sha256

    print(f"Package successfully created: {zip_path}")
    print(f"File count: {file_count} files")
    print(f"Archive SHA-256: {archive_sha}")
    print(f"Archive size: {zip_path.stat().st_size:,} bytes")

    return zip_path, archive_sha, file_count


if __name__ == "__main__":
    create_release_package()
