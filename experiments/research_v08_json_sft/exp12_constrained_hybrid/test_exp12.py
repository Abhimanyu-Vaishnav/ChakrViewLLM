import pytest
import json
import hashlib
from pathlib import Path
from experiments.research_v08_json_sft.exp12_constrained_hybrid.schema_controller import SchemaController

def test_schema_controller_parse_requested_fields():
    controller = SchemaController()
    
    # Single field
    assert controller.parse_requested_fields("Extract the entity.") == ["entity"]
    assert controller.parse_requested_fields("Extract status from context.") == ["status"]
    assert controller.parse_requested_fields("Find the code.") == ["code"]
    
    # Two fields
    fields2 = controller.parse_requested_fields("Extract status and entity from context.")
    assert fields2 == ["entity", "status"]  # canonical ordering
    
    # Three fields
    fields3 = controller.parse_requested_fields("Extract code, status, and entity.")
    assert fields3 == ["entity", "status", "code"]
    
    # Fallback
    fields_all = controller.parse_requested_fields("Return JSON object.")
    assert fields_all == ["entity", "status", "code"]

def test_schema_controller_format_constrained_json():
    controller = SchemaController()
    
    # Single field
    json_str, obj, valid = controller.format_constrained_json(["entity"], {"entity": "solar_inverter"})
    assert valid is True
    assert json_str == '{"entity": "solar_inverter"}'
    assert json.loads(json_str) == {"entity": "solar_inverter"}
    
    # Two fields with unrequested field in dictionary
    json_str2, obj2, valid2 = controller.format_constrained_json(
        ["entity", "status"],
        {"entity": "solar_inverter", "status": "calibrated", "code": "999"}
    )
    assert valid2 is True
    assert json_str2 == '{"entity": "solar_inverter", "status": "calibrated"}'
    assert "code" not in obj2
    
    # Missing value fallback
    json_str3, obj3, valid3 = controller.format_constrained_json(["status"], {})
    assert valid3 is True
    assert obj3 == {"status": "none"}
    assert json_str3 == '{"status": "none"}'

def test_schema_controller_escaping_and_sanitization():
    controller = SchemaController()
    raw_val = '"dangerous" \'value\' \n with whitespace '
    json_str, obj, valid = controller.format_constrained_json(["entity"], {"entity": raw_val})
    assert valid is True
    # Ensure it parses as valid JSON
    parsed = json.loads(json_str)
    assert parsed["entity"] == 'dangerous" \'value\' \n with whitespace'

def test_hash_preservation():
    # Canonical Checkpoint
    canon_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert canon_path.exists()
    canon_hash = hashlib.sha256(canon_path.read_bytes()).hexdigest()
    assert canon_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    
    # Vocab
    vocab_path = Path("data/experiments/vocab_4096/vocab.json")
    vocab_hash = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    assert vocab_hash == "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
    
    # Merges
    merges_path = Path("data/experiments/vocab_4096/merges.json")
    merges_hash = hashlib.sha256(merges_path.read_bytes()).hexdigest()
    assert merges_hash == "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"
    
    # Session Runtime
    session_path = Path("chakrview/runtime/session.py")
    session_hash = hashlib.sha256(session_path.read_bytes()).hexdigest()
    assert session_hash == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    
    # Exp 11 Checkpoint
    exp11_ckpt = Path("experiments/research_v08_json_sft/exp11_hybrid_json/checkpoints/exp11_hybrid_best.pt")
    assert exp11_ckpt.exists()
    exp11_hash = hashlib.sha256(exp11_ckpt.read_bytes()).hexdigest()
    assert exp11_hash == "454fbd1559a686cc30fe5f661f8267a6507d861dccbcda3946987a0044c081e2"

def test_per_example_reproducibility():
    csv_path = Path("experiments/research_v08_json_sft/exp12_constrained_hybrid/EXP12_PER_EXAMPLE.csv")
    json_path = Path("experiments/research_v08_json_sft/exp12_constrained_hybrid/EXP12_RESULTS.json")
    
    assert csv_path.exists()
    assert json_path.exists()
    
    results = json.loads(json_path.read_text(encoding="utf-8"))
    
    # Count metrics directly from CSV
    import csv
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        counts = {
            "val": {"arm_c_json": 0, "arm_c_schema": 0, "arm_c_obj": 0, "total": 0},
            "test_a": {"arm_c_json": 0, "arm_c_schema": 0, "arm_c_obj": 0, "total": 0},
            "test_b": {"arm_c_json": 0, "arm_c_schema": 0, "arm_c_obj": 0, "total": 0},
        }
        for row in reader:
            sp = row["split"]
            counts[sp]["total"] += 1
            if row["arm_c_json_valid"] == "True":
                counts[sp]["arm_c_json"] += 1
            if row["arm_c_schema_valid"] == "True":
                counts[sp]["arm_c_schema"] += 1
            if row["arm_c_exact_obj"] == "True":
                counts[sp]["arm_c_obj"] += 1
                
        for sp in ["val", "test_a", "test_b"]:
            assert counts[sp]["total"] == results[sp]["arm_c_constrained_hybrid"]["total"]
            assert counts[sp]["arm_c_json"] == results[sp]["arm_c_constrained_hybrid"]["json_valid_count"]
            assert counts[sp]["arm_c_schema"] == results[sp]["arm_c_constrained_hybrid"]["schema_valid_count"]
            assert counts[sp]["arm_c_obj"] == results[sp]["arm_c_constrained_hybrid"]["exact_obj_count"]
