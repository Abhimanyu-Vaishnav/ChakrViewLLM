"""
Clean user-space validation smoke test.
Recovered from: experiments/overnight_runs/v08_release_candidate_readiness_20261025_213337/packaging_validation/test_fresh_user_space/ChakrView/smoke_test.py
Validates LocalModelRuntime.from_default() inference in an installed user environment.
"""

import os
import sys
from pathlib import Path

print(f"Current working directory: {os.getcwd()}")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import chakrview
    from chakrview.runtime.local_runtime import LocalModelRuntime
    from chakrview.runtime.inference import GenerationConfig
    print(f"Loaded chakrview from: {chakrview.__file__}")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    runtime = LocalModelRuntime.from_default()
    result = runtime.chat(
        session_id="fresh_user_session",
        prompt="Ram lives in Delhi.",
        tenant_id="fresh_tenant",
        generation_config=GenerationConfig(max_new_tokens=10),
    )
    print("Inference successful!")
    print(f"Generated text: {repr(result.text)}")
    print(f"Tokens: {result.output_token_count}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print(f"Weight Verified: {result.weight_hash_verified}")
except Exception as e:
    print(f"Inference error: {e}")
    sys.exit(2)
