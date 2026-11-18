import pytest
import json
import hashlib
from pathlib import Path
from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e.evaluators.strict_evaluators import StrictEvaluators

REPO_ROOT = Path(r"D:\Project\ChakrView")

class TestUCIP10ER1Comprehensive:

    def test_protected_session_hash_invariance(self):
        p = REPO_ROOT / "chakrview" / "runtime" / "session.py"
        assert p.exists()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"

    def test_checkpoint_hashes_invariance(self):
        targets = {
            "releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt": "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499",
            "releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt": "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0",
            "experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt": "c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830"
        }
        for path_str, exp in targets.items():
            h = hashlib.sha256((REPO_ROOT / path_str).read_bytes()).hexdigest()
            assert h == exp

    def test_manifest_registry_and_suites_existence(self):
        reg_p = REPO_ROOT / "experiments" / "ucip10e" / "manifests" / "manifest_registry.json"
        assert reg_p.exists()
        reg = json.load(open(reg_p, encoding="utf-8"))
        for split, meta in reg["benchmark_suites"].items():
            suite_file = REPO_ROOT / meta["path"]
            assert suite_file.exists()
            assert hashlib.sha256(suite_file.read_bytes()).hexdigest() == meta["sha256"]

    def test_full_840_results_completeness(self):
        res_dir = REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "full_840"
        ckpts = ["chakrview_ucip09_medium_model", "chakrview_ucip09r_medium_model", "ucip09s_balanced_pilot"]
        for c in ckpts:
            p = res_dir / f"eval_full840_{c}.json"
            assert p.exists(), f"Missing 840 result for {c}"
            data = json.load(open(p, encoding="utf-8"))
            assert data["summary"]["total_probes"] == 840
            assert len(data["evaluations"]) == 840

    def test_reproducibility_audit_file(self):
        p = REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "reproducibility" / "reproducibility_audit_30.json"
        assert p.exists()
        data = json.load(open(p, encoding="utf-8"))
        for k in ["ucip09", "ucip09r", "ucip09s"]:
            assert data[k]["deterministic_token_match"] is True

    def test_arithmetic_evaluator_strictness(self):
        probe = {"target_number": 8}
        # True equation
        assert StrictEvaluators.evaluate_arithmetic(probe, "15 - 7 = 8.")["passed"]
        # False equation with operands present
        assert not StrictEvaluators.evaluate_arithmetic(probe, "15 - 8 = 7.")["passed"]

    def test_copying_evaluator_strictness(self):
        probe = {"target_text": "ChakrView operates locally.", "evaluation_type": "verbatim_copy"}
        assert StrictEvaluators.evaluate_copying(probe, "ChakrView operates locally.")["passed"]
        assert not StrictEvaluators.evaluate_copying(probe, "ChakrView operates locally")["passed"] # missing dot fails verbatim
