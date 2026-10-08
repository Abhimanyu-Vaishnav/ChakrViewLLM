"""
Wave 517: Deterministic Release Bundle Generator.
Creates a deterministic release package for ChakrView v0.2.0:
release/
  chakrview-v0.2.0/
    README.md
    CHANGELOG.md
    LICENSE
    CHECKSUMS.sha256
    manifest/
    runtime/
    cognition/
    tokenizer/
    configs/
    benchmarks/
    scripts/
    docs/
"""

import os
import shutil
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "release" / "chakrview-v0.2.0"

FILES_TO_INCLUDE = [
    # Core documents
    ("README.md", "README.md", "Documentation", True),
    ("CHANGELOG.md", "CHANGELOG.md", "Release Notes", True),
    ("LICENSE", "LICENSE", "Legal License", True),

    # Runtime code
    ("chakrview/__init__.py", "chakrview/__init__.py", "Package Root", True),
    ("chakrview/brain/model.py", "chakrview/brain/model.py", "Neural Baseline Core", True),
    ("chakrview/brain/config.py", "chakrview/brain/config.py", "Baseline Configuration", True),
    ("chakrview/runtime/unified_cognitive_runtime.py", "chakrview/runtime/unified_cognitive_runtime.py", "Unified Cognitive Runtime", True),
    ("chakrview/runtime/local_runtime.py", "chakrview/runtime/local_runtime.py", "Local Model Runtime", True),
    ("chakrview/runtime/pipeline.py", "chakrview/runtime/pipeline.py", "Generation Pipeline", True),
    ("chakrview/runtime/sampling.py", "chakrview/runtime/sampling.py", "Sampling Engine", True),

    # Cognition & reasoning
    ("chakrview/cognition/unified_compositional_core.py", "chakrview/cognition/unified_compositional_core.py", "Compositional Relational Core", True),
    ("chakrview/cognition/adaptive_recurrent_reasoning_core.py", "chakrview/cognition/adaptive_recurrent_reasoning_core.py", "Recursive Reasoning Core", True),
    ("chakrview/cognition/neural_routing_gateway.py", "chakrview/cognition/neural_routing_gateway.py", "Neural Routing Gateway", True),
    ("chakrview/cognition/metacognitive_evaluator.py", "chakrview/cognition/metacognitive_evaluator.py", "Metacognitive Evaluator", True),
    ("chakrview/cognition/governed_learning/reasoning_experience.py", "chakrview/cognition/governed_learning/reasoning_experience.py", "Governed RIL Experience", True),
    ("chakrview/cognition/governed_learning/improvement_loop.py", "chakrview/cognition/governed_learning/improvement_loop.py", "Governed Improvement Loop", True),
    ("chakrview/cognition/governed_learning/release_gate.py", "chakrview/cognition/governed_learning/release_gate.py", "Governed Release Gate", True),
    ("chakrview/cognition/distributed/security.py", "chakrview/cognition/distributed/security.py", "Distributed Peering Security", True),
    ("chakrview/cognition/distributed/engine.py", "chakrview/cognition/distributed/engine.py", "Distributed Cognitive Engine", True),

    # Tokenizer
    ("chakrview/tokenizer/tokenizer.py", "chakrview/tokenizer/tokenizer.py", "BPE Tokenizer", True),
    ("data/tokenizer_experiments/v4096/vocab.json", "data/tokenizer_experiments/v4096/vocab.json", "Tokenizer Vocabulary", True),
    ("data/tokenizer_experiments/v4096/merges.json", "data/tokenizer_experiments/v4096/merges.json", "Tokenizer Merges", True),

    # Verification scripts
    ("scripts/verify_release.py", "scripts/verify_release.py", "Release Baseline Verifier", True),

    # Core docs
    ("docs/STEP_513_RELEASE_ARTIFACT_AUDIT.md", "docs/STEP_513_RELEASE_ARTIFACT_AUDIT.md", "Artifact Audit", True),
    ("docs/STEP_514_SECURITY_PROVENANCE_AUDIT.md", "docs/STEP_514_SECURITY_PROVENANCE_AUDIT.md", "Security Audit", True),
    ("docs/STEP_515_DOCUMENTATION_CLAIM_AUDIT.md", "docs/STEP_515_DOCUMENTATION_CLAIM_AUDIT.md", "Documentation Audit", True),
    ("docs/STEP_516_INDEPENDENT_V020_REPRODUCTION.md", "docs/STEP_516_INDEPENDENT_V020_REPRODUCTION.md", "Reproduction Audit", True),
]

def sha256_of_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_bundle():
    print("=" * 60)
    print("GENERATING DETERMINISTIC RELEASE BUNDLE: chakrview-v0.2.0")
    print("=" * 60)

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    checksum_lines = []

    for src_rel, dest_rel, role, required in FILES_TO_INCLUDE:
        src_path = REPO_ROOT / src_rel
        dest_path = BUNDLE_DIR / dest_rel
        
        if not src_path.exists():
            raise FileNotFoundError(f"Missing required release artifact: {src_rel}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        f_hash = sha256_of_file(dest_path)
        f_size = dest_path.stat().st_size

        manifest_entries.append({
            "path": dest_rel,
            "sha256": f_hash,
            "size": f_size,
            "role": role,
            "required": required
        })

        checksum_lines.append(f"{f_hash}  {dest_rel}")
        print(f"[COPIED] {dest_rel:50} | {f_size:7} B | {f_hash[:12]}...")

    # Write CHECKSUMS.sha256
    checksum_file = BUNDLE_DIR / "CHECKSUMS.sha256"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    # Write release_manifest.json
    manifest_file = BUNDLE_DIR / "release_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "release": "chakrview-v0.2.0",
            "runtime_version": "v0.2.0_unified_runtime",
            "artifacts_count": len(manifest_entries),
            "artifacts": manifest_entries
        }, f, indent=2)

    # Verify checksums immediately
    print("\nVerifying generated CHECKSUMS.sha256...")
    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            expected_hash, rel_f = line.strip().split("  ")
            actual_hash = sha256_of_file(BUNDLE_DIR / rel_f)
            assert expected_hash == actual_hash, f"Checksum mismatch for {rel_f}!"

    print("Checksum verification: ALL OK (Deterministic match).")
    print(f"Bundle successfully created at: {BUNDLE_DIR}")

if __name__ == "__main__":
    generate_bundle()
