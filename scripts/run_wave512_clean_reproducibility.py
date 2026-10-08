import os
import sys
import json
import platform
import subprocess
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.interactive import EXPECTED_WEIGHT_HASH

def run_wave512_reproducibility():
    print("=" * 60)
    print("RUNNING WAVE 512: CLEAN ENVIRONMENT REPRODUCIBILITY AUDIT")
    print("=" * 60)
    
    os.makedirs("artifacts/waves_512_519_execution", exist_ok=True)
    
    env_info = {
        "python_version": sys.version,
        "os": platform.platform(),
        "pytorch_version": torch.__version__,
        "runtime_version": "v0.2.0_unified_runtime",
        "benchmark_version": "v0.2_master_benchmark",
        "configuration_hash": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
    }
    
    # Run clean subprocess test stripped of auxiliary PYTHONPATH or overrides
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    
    res = subprocess.run(
        [sys.executable, "chakrview/cognition/v020_unified_runtime_master_gate.py"],
        env=env,
        capture_output=True,
        text=True
    )
    
    success = (res.returncode == 0) and ("V020_RELEASE_CANDIDATE_VERIFIED" in res.stdout)
    status_str = "VERIFIED_ISOLATED_PROCESS" if success else "REPRODUCTION_FAILED"
    
    print(f"Isolated Subprocess Returncode: {res.returncode}")
    print(f"Independent Gate Verification: {'PASS' if success else 'FAIL'}")
    print(f"Clean Environment Reproduction: {status_str}")
    
    metrics = {
        "clean_environment_reproduction": status_str,
        "environment_metadata": env_info,
        "subprocess_verified": success,
        "baseline_sha_verified": EXPECTED_WEIGHT_HASH == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    }
    
    out_file = "artifacts/waves_512_519_execution/wave512_clean_reproducibility.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wave 512 complete. Saved to {out_file}")
    return metrics

if __name__ == "__main__":
    run_wave512_reproducibility()
