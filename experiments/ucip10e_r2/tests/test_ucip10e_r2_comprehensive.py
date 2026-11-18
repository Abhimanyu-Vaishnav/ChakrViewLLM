import pytest
import json
import hashlib
from pathlib import Path
from experiments.ucip10e_r2.evaluators.strict_evaluators_r2 import StrictEvaluatorsR2

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e_r2"

class TestUCIP10ER2Comprehensive:

    def test_protected_session_hash(self):
        p = REPO_ROOT / "chakrview" / "runtime" / "session.py"
        assert p.exists()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"

    def test_checkpoint_hashes(self):
        targets = {
            "releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt": "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499",
            "releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt": "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0",
            "experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt": "c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830"
        }
        for path_str, exp in targets.items():
            h = hashlib.sha256((REPO_ROOT / path_str).read_bytes()).hexdigest()
            assert h == exp

    def test_r2_manifest_and_zero_contamination(self):
        reg = json.load(open(BASE_DIR / "manifests" / "manifest_registry_r2.json", encoding="utf-8"))
        dev_p = REPO_ROOT / reg["benchmark_suites"]["r2_development"]["path"]
        sealed_p = REPO_ROOT / reg["benchmark_suites"]["r2_sealed"]["path"]
        assert dev_p.exists() and sealed_p.exists()
        
        dev_data = json.load(open(dev_p, encoding="utf-8"))
        sealed_data = json.load(open(sealed_p, encoding="utf-8"))
        assert len(dev_data) == 460
        assert len(sealed_data) == 460

        dev_prompts = set(p["prompt"].strip().lower() for p in dev_data)
        sealed_prompts = set(p["prompt"].strip().lower() for p in sealed_data)
        assert len(dev_prompts.intersection(sealed_prompts)) == 0, "Dev and Sealed must be 100% disjoint!"

    def test_r2_raw_eval_results_completeness(self):
        for sub, name in [("uc09", "chakrview_ucip09_medium_model"), ("uc09r", "chakrview_ucip09r_medium_model"), ("uc09s", "ucip09s_balanced_pilot")]:
            res_p = BASE_DIR / "results" / sub / f"eval_r2_{name}.json"
            assert res_p.exists(), f"Missing R2 result file {res_p}"
            data = json.load(open(res_p, encoding="utf-8"))
            assert data["summary"]["total_probes"] == 460
            assert len(data["evaluations"]) == 460

    def test_reproducibility_audit_r2(self):
        repro_p = BASE_DIR / "results" / "reproducibility" / "reproducibility_r2_audit_30.json"
        assert repro_p.exists()
        data = json.load(open(repro_p, encoding="utf-8"))
        for k in ["ucip09", "ucip09r", "ucip09s"]:
            assert data[k]["deterministic_token_match"] is True

    def test_cyclic_phrase_detector_integrity(self):
        probe = {"subdomain": "english", "expected_keywords": ["cpu", "memory"]}
        cyclic_text = "cpu memory processor cpu memory processor cpu memory processor"
        assert not StrictEvaluatorsR2.evaluate_language_r2(probe, cyclic_text)["passed"]
