import pytest
import json
import hashlib
from pathlib import Path
from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e.runners.eval_engine import evaluate_probe

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e"

class TestUCIP10EInfrastructure:

    def test_protected_session_hash(self):
        p = REPO_ROOT / "chakrview" / "runtime" / "session.py"
        assert p.exists()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"

    def test_baseline_checkpoint_hashes(self):
        targets = {
            "releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt": "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499",
            "releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt": "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0",
            "experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt": "c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830"
        }
        for path_str, expected in targets.items():
            p = REPO_ROOT / path_str
            assert p.exists()
            assert hashlib.sha256(p.read_bytes()).hexdigest() == expected

    def test_manifest_registry_integrity(self):
        reg = json.load(open(BASE_DIR / "manifests" / "manifest_registry.json", encoding="utf-8"))
        for split, meta in reg["benchmark_suites"].items():
            p = REPO_ROOT / meta["path"]
            assert p.exists()
            assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]

    def test_probe_schema_fields(self):
        suite = json.load(open(BASE_DIR / "datasets" / "sealed" / "benchmark_sealed_core100.json", encoding="utf-8"))
        assert len(suite) == 100
        required_keys = {"id", "domain", "prompt", "evaluation_type", "difficulty", "split", "provenance"}
        for probe in suite:
            assert required_keys.issubset(probe.keys())

    def test_model_adapter_loading_and_generation(self):
        ckpt = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
        adapter = ModelAdapter(ckpt)
        assert adapter.param_count == 12932736
        out = adapter.generate("Hello ChakrView. How are you?", max_new_tokens=16)
        assert len(out.strip()) > 0
        assert "ChakrView" in out

    def test_evaluate_probe_dispatcher(self):
        probe = {
            "domain": "mathematics",
            "evaluation_type": "exact_math",
            "target_number": 7
        }
        assert evaluate_probe(probe, "3 + 4 = 7.")["passed"]
        assert not evaluate_probe(probe, "3 + 4 = 8.")["passed"]
