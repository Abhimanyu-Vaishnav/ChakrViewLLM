import pytest
from chakrview.data.coding_pipeline import CodingDataPipeline, CodingRecord


def test_coding_record_validation():
    valid_data = {
        "task_id": "CODE-001",
        "task_type": "code_continuation",
        "language": "python",
        "prompt": "def foo():\n    return ",
        "response": "42",
        "provenance": "synthetic_verified",
        "split": "train"
    }
    rec = CodingRecord.from_dict(valid_data)
    assert rec.task_id == "CODE-001"
    assert rec.task_type == "code_continuation"
    
    # Missing required field
    invalid_data = valid_data.copy()
    del invalid_data["prompt"]
    with pytest.raises(ValueError, match="Missing required field"):
        CodingRecord.from_dict(invalid_data)

    # Invalid task type
    bad_task = valid_data.copy()
    bad_task["task_type"] = "unsupported_task_type"
    with pytest.raises(ValueError, match="Invalid task_type"):
        CodingRecord.from_dict(bad_task)


def test_pipeline_duplicate_and_leakage_detection(tmp_path):
    pipeline = CodingDataPipeline(max_seq_len=128)
    records = [
        {
            "task_id": "T1",
            "task_type": "code_continuation",
            "language": "python",
            "prompt": "def add(a, b): return ",
            "response": "a + b",
            "provenance": "unit_test",
            "split": "train",
            "template_family": "math_ops"
        },
        {
            "task_id": "T2",
            "task_type": "code_continuation",
            "language": "python",
            "prompt": "def add(a, b): return ", # Exact duplicate
            "response": "a + b",
            "provenance": "unit_test",
            "split": "train",
            "template_family": "math_ops"
        },
        {
            "task_id": "T3",
            "task_type": "code_continuation",
            "language": "python",
            "prompt": "def add_two(x, y): return ",
            "response": "a + b", # Target leakage into val!
            "provenance": "unit_test",
            "split": "val",
            "template_family": "math_ops" # Family overlap
        }
    ]
    report = pipeline.validate_dataset(records)
    assert report["total_records"] == 3
    assert report["valid_records"] == 3
    assert report["exact_duplicate_prompts"] == 1
    assert report["target_leakage_cases"] >= 1
    assert report["template_family_overlap"] >= 1
    assert report["leakage_safe"] is False

    # Export manifest
    out_file = tmp_path / "test_coding_dataset.jsonl"
    manifest = pipeline.export_dataset_manifest(str(out_file), records)
    assert manifest["record_count"] == 3
    assert len(manifest["sha256"]) == 64
    assert out_file.exists()
