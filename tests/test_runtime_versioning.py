"""
Unit tests for Step 9 Versioning & Lineage Layer (chakrview.runtime.versioning).
"""

import pytest
from pathlib import Path
from chakrview.runtime.versioning import (
    BrainProfileType,
    BrainVersionManifest,
    VersionLineageTracker,
)


def test_manifest_serialization_and_lineage():
    manifest = BrainVersionManifest(
        version_id="chakrview-v0.1.0-universal",
        profile_type=BrainProfileType.UNIVERSAL,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        tokenizer_checksum="7498d92adeef7c6db98d89a444a7f0e303dd5e7ea4b679a95781a95e6347c617",
        skills=["skill_general_v1"],
        knowledge_indexes=["idx_stage_c_v1"],
    )
    # version_id is automatically added to lineage
    assert "chakrview-v0.1.0-universal" in manifest.lineage

    json_str = manifest.to_json()
    restored = BrainVersionManifest.from_json(json_str)
    assert restored.version_id == manifest.version_id
    assert restored.profile_type == BrainProfileType.UNIVERSAL
    assert restored.base_model_hash == manifest.base_model_hash


def test_lineage_tracker_hierarchy(tmp_path: Path):
    tracker = VersionLineageTracker(storage_dir=tmp_path)

    # 1. Root Universal Version
    root_manifest = BrainVersionManifest(
        version_id="chakrview-v0.1.0-universal",
        profile_type=BrainProfileType.UNIVERSAL,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_v01",
        tokenizer_checksum="hash_tok_v01",
    )
    tracker.register_version(root_manifest)
    assert tracker.count() == 1

    # 2. Derived Company-A Version
    company_manifest = BrainVersionManifest(
        version_id="chakrview-v0.1.0-companyA-v0.1",
        profile_type=BrainProfileType.ENTERPRISE,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_v01",
        tokenizer_checksum="hash_tok_v01",
        parent_version_id="chakrview-v0.1.0-universal",
        lineage=["chakrview-v0.1.0-universal"],
        knowledge_indexes=["companyA_docs_idx"],
    )
    tracker.register_version(company_manifest)
    assert tracker.count() == 2

    # 3. Derived User-Specific Version from Company-A
    user_manifest = BrainVersionManifest(
        version_id="chakrview-v0.1.0-companyA-userBob-v0.1",
        profile_type=BrainProfileType.USER_SPECIFIC,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_v01",
        tokenizer_checksum="hash_tok_v01",
        parent_version_id="chakrview-v0.1.0-companyA-v0.1",
        lineage=["chakrview-v0.1.0-universal", "chakrview-v0.1.0-companyA-v0.1"],
    )
    tracker.register_version(user_manifest)
    assert tracker.count() == 3

    # Check ancestry traversal
    chain = tracker.get_lineage("chakrview-v0.1.0-companyA-userBob-v0.1")
    assert len(chain) == 3
    assert chain[0].version_id == "chakrview-v0.1.0-universal"
    assert chain[1].version_id == "chakrview-v0.1.0-companyA-v0.1"
    assert chain[2].version_id == "chakrview-v0.1.0-companyA-userBob-v0.1"


def test_lineage_tracker_rejects_broken_parent():
    tracker = VersionLineageTracker()
    manifest = BrainVersionManifest(
        version_id="chakrview-orphaned-v0.1",
        profile_type=BrainProfileType.ENTERPRISE,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_v01",
        tokenizer_checksum="hash_tok_v01",
        parent_version_id="non_existent_parent",
    )
    # Must fail because parent does not exist
    with pytest.raises(ValueError, match="not found in registry"):
        tracker.register_version(manifest)


def test_lineage_tracker_rejects_incompatible_base_model():
    tracker = VersionLineageTracker()
    root = BrainVersionManifest(
        version_id="root_v1",
        profile_type=BrainProfileType.UNIVERSAL,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_A",
        tokenizer_checksum="hash_tok_A",
    )
    tracker.register_version(root)

    # Derived tries to claim root as parent, but has a different base model hash!
    derived = BrainVersionManifest(
        version_id="derived_v1",
        profile_type=BrainProfileType.ENTERPRISE,
        base_model_version="chakrmicro-v0.1",
        base_model_hash="hash_base_DIFFERENT",
        tokenizer_checksum="hash_tok_A",
        parent_version_id="root_v1",
    )
    with pytest.raises(ValueError, match="Lineage verification failed"):
        tracker.register_version(derived)
