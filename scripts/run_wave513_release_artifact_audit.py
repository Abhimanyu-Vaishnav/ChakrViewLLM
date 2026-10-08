import os
import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_wave513_release_artifact_audit():
    print("=" * 60)
    print("RUNNING WAVE 513: RELEASE ARTIFACT INVENTORY AUDIT")
    print("=" * 60)
    
    repo_root = Path(__file__).resolve().parent.parent
    
    # Audit primary release candidate artifacts across the repository
    core_artifacts = [
        ("chakrview/__init__.py", "core_module", "Mandatory package entrypoint"),
        ("chakrview/brain/model.py", "core_neural_architecture", "ChakrMicro 3.44M canonical neural brain"),
        ("chakrview/brain/config.py", "core_config", "ModelConfig hyperparameters"),
        ("chakrview/cognition/unified_compositional_core.py", "cognitive_core", "UnifiedCompositionalCore sequence core"),
        ("chakrview/cognition/adaptive_recurrent_reasoning_core.py", "recurrent_core", "Adaptive recurrent reasoning core"),
        ("chakrview/cognition/neural_routing_gateway.py", "routing_gateway", "NeuralCognitiveRoutingGateway (74k params)"),
        ("chakrview/cognition/metacognitive_evaluator.py", "metacognition", "MetaCognitiveEvaluator and strategy selector"),
        ("chakrview/cognition/governed_learning/reasoning_experience.py", "experience_schema", "ReasoningExperience schema"),
        ("chakrview/runtime/unified_cognitive_runtime.py", "runtime_engine", "v0.2.0 Unified Cognitive Runtime engine"),
        ("chakrview/tokenizer/tokenizer.py", "tokenizer_core", "Byte-level BPE tokenizer"),
        ("data/tokenizer_experiments/v4096/vocab.json", "tokenizer_vocab", "Vocabulary mapping"),
        ("data/tokenizer_experiments/v4096/merges.json", "tokenizer_merges", "BPE merge table"),
        ("scripts/verify_release.py", "release_verifier", "One-command canonical baseline release verifier"),
        ("LICENSE", "legal", "Apache-2.0 / Project License"),
        ("README.md", "documentation", "Project overview and verified capability contract")
    ]
    
    inventory = []
    missing = []
    
    for rel_path, art_type, purpose in core_artifacts:
        full_path = repo_root / rel_path
        if full_path.exists():
            digest = sha256_file(full_path)
            sz = full_path.stat().st_size
            inventory.append({
                "path": rel_path,
                "type": art_type,
                "purpose": purpose,
                "size_bytes": sz,
                "sha256": digest,
                "required": True
            })
            print(f"[FOUND] {rel_path:<48} | {sz:>7} B | {digest[:12]}...")
        else:
            missing.append(rel_path)
            print(f"[MISSING] {rel_path}")
            
    all_present = (len(missing) == 0)
    print(f"\nRelease Artifact Inventory Audit: {'PASS' if all_present else 'FAIL'} ({len(inventory)}/{len(core_artifacts)})")
    
    out_data = {
        "inventory": inventory,
        "missing": missing,
        "total_audited": len(core_artifacts),
        "audit_passed": all_present
    }
    
    out_file = repo_root / "artifacts/waves_512_519_execution/wave513_artifact_inventory.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wave 513 complete. Saved to {out_file}")
    return out_data

if __name__ == "__main__":
    run_wave513_release_artifact_audit()
