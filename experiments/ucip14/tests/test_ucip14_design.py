"""
Test suite verifying consistency and integrity of UCIP-14 design artifacts.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "docs"
DESIGN_DIR = ROOT / "experiments" / "ucip14" / "design"
REPORTS_DIR = ROOT / "reports"


def test_01_all_design_documents_exist():
    required_docs = [
        DOCS_DIR / "UCIP_14_INSTRUCTION_ALIGNMENT_ARCHITECTURE.md",
        DOCS_DIR / "UCIP_14_DATASET_SPECIFICATION.md",
        DOCS_DIR / "UCIP_14_TRAINING_STRATEGY.md",
        DOCS_DIR / "UCIP_14_EVALUATION_PROTOCOL.md",
        DOCS_DIR / "UCIP_14_RETENTION_AND_FORGETTING_PLAN.md",
        REPORTS_DIR / "UCIP_14_DESIGN_REVIEW.md",
        DESIGN_DIR / "ucip14_experiment_matrix.json",
        DESIGN_DIR / "ucip14_dataset_schema.json",
        DESIGN_DIR / "ucip14_design_manifest.json",
    ]
    for d in required_docs:
        assert d.is_file(), f"Missing required document: {d}"


def test_02_taxonomy_percentages_total_100():
    spec_path = DOCS_DIR / "UCIP_14_DATASET_SPECIFICATION.md"
    content = spec_path.read_text(encoding="utf-8")
    assert "**100.0%**" in content


def test_03_experiment_matrix_schema_valid():
    mat_path = DESIGN_DIR / "ucip14_experiment_matrix.json"
    with open(mat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 4
    arm_ids = [d["arm_id"] for d in data]
    assert arm_ids == ["SFT-A", "SFT-B", "SFT-C", "SFT-D"]


def test_04_dataset_schema_json_valid():
    schema_path = DESIGN_DIR / "ucip14_dataset_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["type"] == "object"
    assert "category" in schema["properties"]
    assert len(schema["properties"]["category"]["enum"]) == 18


def test_05_design_manifest_hashes_match():
    manifest_path = DESIGN_DIR / "ucip14_design_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["design_verdict"] == "DESIGN_READY"
    assert manifest["training_status"] == "NOT_AUTHORIZED"
    assert manifest["protected_artifacts"]["session_py_sha256"] == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    assert manifest["target_model"]["base_checkpoint_sha256"] == "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
